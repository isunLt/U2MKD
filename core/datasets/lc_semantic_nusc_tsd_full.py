import os
import numpy as np
from functools import reduce

import torch
from torch.utils import data

from nuscenes import NuScenes as NuScenes_devkit
from nuscenes.utils.data_classes import transform_matrix
from torchvision import transforms as T

from torchpack.environ import get_run_dir
from torchpack.utils.config import configs

from core.datasets.utils import PCDTransformTool, InstAugmentationV2
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from PIL import Image
from pyquaternion import Quaternion
from copy import deepcopy

CLASS_WEIGHT = [
    0,
    10.6310,
    29.8912,
    14.3095,
    5.4934,
    20.9552,
    27.1116,
    18.4044,
    24.8379,
    13.9495,
    8.3447,
    1.9305,
    11.0304,
    4.0755,
    4.0729,
    2.5711,
    3.0951
]

NUSCENES_MAP_LABEL2NAME_16 = {
    0: 'noise',
    1: 'barrier',
    2: 'bicycle',
    3: 'bus',
    4: 'car',
    5: 'construction_vehicle',
    6: 'motorcycle',
    7: 'pedestrian',
    8: 'traffic_cone',
    9: 'trailer',
    10: 'truck',
    11: 'driveable_surface',
    12: 'other_flat',
    13: 'sidewalk',
    14: 'terrain',
    15: 'manmade',
    16: 'vegetation',
}


class LCNuScenesTSDistillFull(dict):
    def __init__(self, root, voxel_size, version, verbose):
        self.nusc = NuScenes_devkit(dataroot=root, version=version, verbose=verbose)
        super(LCNuScenesTSDistillFull, self).__init__({
            "train": _LCNuScenesTSDistillFullInternal(nusc=self.nusc, voxel_size=voxel_size, split="train"),
            "val": _LCNuScenesTSDistillFullInternal(nusc=self.nusc, voxel_size=voxel_size, split="val")
        })


class _LCNuScenesTSDistillFullInternal(data.Dataset):
    labels_mapping = {
        1: 0,
        5: 0,
        7: 0,
        8: 0,
        10: 0,
        11: 0,
        13: 0,
        19: 0,
        20: 0,
        0: 0,
        29: 0,
        31: 0,
        9: 1,
        14: 2,
        15: 3,
        16: 3,
        17: 4,
        18: 5,
        21: 6,
        2: 7,
        3: 7,
        4: 7,
        6: 7,
        12: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        30: 16
    }

    CAM_CHANNELS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    IMAGE_SIZE = (900, 1600)

    CLASS_DISTRIBUTE = [2.87599770e-01, 7.84806427e-03, 1.19217528e-04, 3.88372281e-03, 3.21376629e-02, 1.27727921e-03,
                        3.60467902e-04, 1.95227505e-03, 6.20954881e-04, 4.13906749e-03, 1.33608580e-02, 2.67327832e-01,
                        7.21896959e-03, 5.92055787e-02, 5.92833998e-02, 1.50278018e-01, 1.03386862e-01]

    CLASS_NAME = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                  'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',  'truck',]

    INST_DATABASE_DISTRIBUTE = [0, 60272, 2561, 8091, 156414, 6908, 3036, 35011, 13188, 14186, 41250]

    def __init__(self, nusc, voxel_size, split):
        self.nusc = nusc
        self.voxel_size = voxel_size
        self.split = split
        self.ignored_labels = configs['criterion']['ignore_index']
        self.run_dir = get_run_dir()
        self.num_classes = configs['data']['num_classes']
        self.flip_aug = configs['dataset']['flip']
        # self.bottom_crop = configs['dataset']['bottom_crop']
        # self.color_jitter = T.ColorJitter(*configs['dataset']['color_jitter'])
        im_cr = configs['dataset']['im_cr']
        self.input_image_size = [int(x * im_cr) for x in self.IMAGE_SIZE]
        print('input_image_size:', self.input_image_size)
        self.transform = T.Compose([T.Resize(size=self.input_image_size)])
        self.debug = configs['debug']['debug_val']
        self.im_drop = configs['dataset']['im_drop']
        self.inst_aug = configs['dataset']['inst_aug']
        if self.inst_aug:
            inst_pkl_path = configs['dataset']['inst_pkl_path']
            thing_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint8)
            ground_list = np.array([11, 12, 13], dtype=np.uint8)
            pair_list = [[11], [11], [11], [11], [11], [11], [11, 12, 13], [11, 12, 13], [11], [11]]
            inst_add_num = configs['dataset']['inst_add_num']
            self.inst_augmenter = InstAugmentationV2(instance_pkl_path=inst_pkl_path, thing_list=thing_list,
                                                     ground_list=ground_list, pair_list=pair_list, add_num=inst_add_num,
                                                     num_classes=self.num_classes, class_name=self.CLASS_NAME,
                                                     class_weight=self.INST_DATABASE_DISTRIBUTE,
                                                     class_min_num=[10, 10, 100, 100, 100, 10, 10, 10, 100, 100],
                                                     random_flip=False, random_rotate=False, random_trans=True,
                                                     feat_dim=4)
        self.multisweeps = configs['dataset']['multisweeps']['num_sweeps']
        self.only_past = configs['dataset']['multisweeps']['only_past']

        if self.split == "train":
            select_idx = np.load("./data/nuscenes/nuscenes_train_official.npy")
            self.sample = [self.nusc.sample[i] for i in select_idx]
        elif self.split == "val":
            select_idx = np.load("./data/nuscenes/nuscenes_val_official.npy")
            self.sample = [self.nusc.sample[i] for i in select_idx]
        elif self.split == "test":
            self.sample = self.nusc.sample
        else:
            print("split not implement yet, exit!")
            exit(-1)

    def __len__(self):
        'Denotes the total number of samples'
        # return 10
        return len(self.sample)

    def _rotate_and_scale(self, pts):
        pts_cp = np.zeros_like(pts)
        theta = np.random.uniform(0, 2 * np.pi)
        scale_factor = np.random.uniform(0.95, 1.05)
        rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                            [-np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
        pts_cp[:, :3] = np.dot(pts[:, :3], rot_mat) * scale_factor
        pts_cp[:, 3] = pts[:, 3]
        return pts_cp

    def _filp3d(self, pts):
        flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            pts[:, 0] = -pts[:, 0]
        elif flip_type == 2:
            pts[:, 1] = -pts[:, 1]
        elif flip_type == 3:
            pts[:, :2] = -pts[:, :2]
        return pts

    def _process_unimodal_input(self, pts, labels_, sample):

        pts = deepcopy(pts)
        if self.multisweeps != 0:
            agg_pts, agg_ts = self._aggregate_lidar_sweeps(sample_ref=sample, nsweeps=self.multisweeps, only_past=self.only_past)
            agg_pts = np.concatenate([pts] + agg_pts, axis=0)
            agg_ts = np.concatenate([np.zeros(shape=(pts.shape[0],))] + agg_ts, axis=0)
            keyframe_mask = (agg_ts == 0)
            num = np.sum(~keyframe_mask)
            labels_ = np.concatenate([labels_, np.full(shape=(int(num),), fill_value=self.ignored_labels, dtype=np.uint8)], axis=0)
            pts = agg_pts

        if 'train' in self.split:
            pts = self._rotate_and_scale(pts)
            pts = self._filp3d(pts)
        voxel = np.round(pts[:, :3] / self.voxel_size).astype(np.int32)
        voxel -= voxel.min(0, keepdims=1)
        feat_ = pts.astype(np.float32)

        # inds: voxel使用散列方程key=(x*y_max+y)*z_max+z后, np.unique(key)的结果, 也就是重复网格坐标只取第一个遇到的
        _, inds, inverse_map = sparse_quantize(voxel,
                                               return_index=True,
                                               return_inverse=True)

        voxel_full = voxel[inds]
        feat_full = feat_[inds]
        labels_full = labels_[inds]

        lidar = SparseTensor(feat_full, voxel_full)
        labels = SparseTensor(labels_full, voxel_full)
        labels_ = SparseTensor(labels_, voxel)
        inverse_map = SparseTensor(inverse_map, voxel)

        feed_dict = {
            'lidar': lidar,
            'targets': labels,
            "targets_mapped": labels_,
            "inverse_map": inverse_map,
            'num_vox': voxel_full.shape[0],
            'num_pts': voxel.shape[0]
        }

        if self.multisweeps != 0:
            feed_dict['keyframe_mask'] = SparseTensor(keyframe_mask[inds], voxel_full)
            feed_dict['keyframe_mask_full'] = SparseTensor(keyframe_mask, voxel)
        return feed_dict

    def _aggregate_lidar_sweeps(self, sample_ref, nsweeps, only_past=False):

        def _remove_close(xyz: np.ndarray, min_dist):
            x_mask = np.fabs(xyz[:, 0]) < min_dist
            y_mask = np.fabs(xyz[:, 1]) < min_dist
            return np.logical_and(x_mask, y_mask)

        def _agg_sweeps(sweeps_num, direction='prev'):
            assert direction in ['prev', 'next']
            current_sd_rec = ref_sd_rec
            pts = []
            ts = []
            for _ in range(sweeps_num):
                if current_sd_rec[direction] == '':
                    break
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec[direction])
                # Load up the pointcloud and remove points close to the sensor.
                curr_pts_path = os.path.join(self.nusc.dataroot, current_sd_rec['filename'])
                curr_pts = np.fromfile(curr_pts_path, dtype=np.float32).reshape([-1, 5])[:, :4]
                close_mask = _remove_close(curr_pts, min_dist=1.0)
                curr_pts = curr_pts[~close_mask]
                pcd_trans_tool = PCDTransformTool(curr_pts)

                # Get past pose.
                current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'],
                                                    Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                pcd_trans_tool.transform(trans_matrix)

                # Add time vector which can be used as a temporal feature.
                if direction == 'prev':
                    time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
                else:
                    time_lag = 1e-6 * current_sd_rec['timestamp'] - ref_time  # Positive difference.
                times = time_lag * np.ones((curr_pts.shape[0],))
                ts.append(times)
                transformed_pts = np.concatenate([pcd_trans_tool.pcd.T, curr_pts[:, 3].reshape(-1, 1)], axis=-1)
                pts.append(transformed_pts)
            return pts, ts

        # Get reference pose and timestamp.
        ref_sd_token = sample_ref['data']['LIDAR_TOP']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        prev_pts, prev_ts = _agg_sweeps(sweeps_num=nsweeps, direction='prev')
        if not only_past:
            next_pts, next_ts = _agg_sweeps(sweeps_num=(2*nsweeps-len(prev_pts)), direction='next')
        else:
            next_pts, next_ts = [], []

        return prev_pts + next_pts, prev_ts + next_ts

    def __getitem__(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        if self.split == "test":
            labels_raw = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            lidar_label_path = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"])
            labels_raw = np.fromfile(lidar_label_path, dtype=np.uint8).reshape([-1, 1])
            labels_raw = np.vectorize(self.labels_mapping.__getitem__)(labels_raw).flatten()

        if 'train' in self.split and self.inst_aug:
            raw_pts_num = pts.shape[0]
            pts, labels_raw, intensity = self.inst_augmenter.inst_aug(pts[:, :3], labels_raw, pts[:, 3:])
            pts = np.concatenate([pts, intensity], axis=-1)
            # if np.random.random(1) < 0.5:
            #     pts, labels_raw, intensity = self.inst_augmenter_add.inst_aug(pts[:, :3], labels_raw, pts[:, 3:])
            #     pts = np.concatenate([pts, intensity], axis=-1)
            labels_raw = labels_raw.flatten()
            inst_aug_mask = np.zeros_like(labels_raw).astype(bool)
            inst_aug_mask[:raw_pts_num] = True

        feed_dict_t = self._process_unimodal_input(pts, labels_raw, sample)

        camera_channel = []
        pixel_coordinates = []  # 6, N, 2
        masks = []
        valid_mask = np.array([-1] * pts.shape[0])
        im_drop_idx = np.random.choice(len(self.CAM_CHANNELS), self.im_drop, replace=False)

        for idx, channel in enumerate(self.CAM_CHANNELS):
            if 'train' in self.split and idx in im_drop_idx:
                continue
            cam_token = sample['data'][channel]
            cam_channel = self.nusc.get('sample_data', cam_token)
            im = Image.open(os.path.join(self.nusc.dataroot, cam_channel['filename'])).convert('RGB')
            camera_channel.append(np.array(self.transform(im)))
            pcd_trans_tool = PCDTransformTool(pts[:, :3])
            # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = self.nusc.get('calibrated_sensor', lidar_channel['calibrated_sensor_token'])
            pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(cs_record['translation']))
            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get('ego_pose', lidar_channel['ego_pose_token'])
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(poserecord['translation']))
            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = self.nusc.get('ego_pose', cam_channel['ego_pose_token'])
            pcd_trans_tool.translate(-np.array(poserecord['translation']))
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get('calibrated_sensor', cam_channel['calibrated_sensor_token'])
            pcd_trans_tool.translate(-np.array(cs_record['translation']))
            pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
            mask = np.ones(pts.shape[0], dtype=bool)
            mask = np.logical_and(mask, pcd_trans_tool.pcd[2, :] > 1)
            # Fifth step: project from 3d coordinate to 2d coordinate
            pcd_trans_tool.pcd2image(np.array(cs_record['camera_intrinsic']))
            pixel_coord = pcd_trans_tool.pcd[:2, :]
            pixel_coord[0, :] = pixel_coord[0, :] / (im.size[0] - 1.0) * 2.0 - 1.0  # width
            pixel_coord[1, :] = pixel_coord[1, :] / (im.size[1] - 1.0) * 2.0 - 1.0  # height
            # pixel_coordinates.append(pixel_coord.T)

            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            mask = np.logical_and(mask, pixel_coord[0, :] > -1)
            mask = np.logical_and(mask, pixel_coord[0, :] < 1)
            mask = np.logical_and(mask, pixel_coord[1, :] > -1)
            mask = np.logical_and(mask, pixel_coord[1, :] < 1)
            valid_mask[mask] = idx
            masks.append(mask)
            pixel_coordinates.append(pixel_coord.T)

        pt_with_img_idx = (valid_mask != -1)
        pixel_coordinates = np.stack(pixel_coordinates, axis=0)
        masks = np.stack(masks, axis=0)
        camera_channel = np.stack(camera_channel, axis=0)

        if 'train' in self.split and self.inst_aug:
            pt_with_img_idx = np.logical_and(pt_with_img_idx, inst_aug_mask)
            masks = np.logical_and(masks, inst_aug_mask)

        pts_cp = np.zeros_like(pts)
        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
            pts_cp[:, :3] = np.dot(pts[:, :3], rot_mat) * scale_factor
        else:
            theta = 0.0
            transform_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                      [-np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 1]])
            pts_cp[...] = pts[...]
            pts_cp[:, :3] = np.dot(pts_cp[:, :3], transform_mat)

        pts_cp[:, 3] = pts[:, 3]  # block为经过随机旋转和放缩的点云 [N, 4] -> x,y,z,sig
        voxel = np.round(pts_cp[:, :3] / self.voxel_size).astype(np.int32)  # voxelization
        voxel -= voxel.min(0, keepdims=1)  # 将体素坐标最小值为0

        feat_ = pts_cp.astype(np.float32)

        # inds: voxel使用散列方程key=(x*y_max+y)*z_max+z后, np.unique(key)的结果, 也就是重复网格坐标只取第一个遇到的
        _, inds, inverse_map = sparse_quantize(voxel,
                                               return_index=True,
                                               return_inverse=True)

        voxel_full = voxel[inds]
        feat_full = feat_[inds]
        labels_full = labels_raw[inds]
        masks = masks[:, inds]
        pixel_coordinates = pixel_coordinates[:, inds, :]
        lidar = SparseTensor(feat_full, voxel_full)
        labels = SparseTensor(labels_full, voxel_full)
        labels_ = SparseTensor(labels_raw, voxel)
        inverse_map = SparseTensor(inverse_map, voxel)
        fov_mask = SparseTensor(pt_with_img_idx[inds], voxel_full)

        feed_dict_s = {
            'lidar': lidar,
            'targets': labels,
            "targets_mapped": labels_,
            "inverse_map": inverse_map,
            'images': camera_channel,
            "pixel_coordinates": pixel_coordinates,  # [6, N, 2]
            "masks": masks,  # [6, N]
            'fov_mask': fov_mask,
            'inds': [inds],
            'num_vox': voxel_full.shape[0]
            # 'targets_mapped_fov': labels_fov
        }

        if 'train' in self.split and self.inst_aug:
            feed_dict_s['inst_aug_mask'] = SparseTensor(inst_aug_mask[inds], voxel_full)

        if self.debug:
            label_fov = np.full_like(labels_raw, fill_value=self.ignored_labels, dtype=np.uint8)
            label_fov[pt_with_img_idx] = labels_raw[pt_with_img_idx]
            feed_dict_s['label_fov'] = SparseTensor(label_fov, voxel)

        return {
            'feed_dict_s': feed_dict_s,
            'feed_dict_t': feed_dict_t,
            'lidar_token': lidar_token,
        }

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], dict):
            ans_dict = {}
            for key in batch[0].keys():
                if key == "masks":
                    ans_dict[key] = [torch.from_numpy(sample[key]) for sample in batch]
                elif key == "pixel_coordinates":
                    ans_dict[key] = [torch.from_numpy(sample[key]).float() for sample in batch]
                elif isinstance(batch[0][key], SparseTensor):
                    ans_dict[key] = sparse_collate(
                        [sample[key] for sample in batch])  # sparse_collate_tensor -> sparse_collate
                elif isinstance(batch[0][key], np.ndarray):
                    ans_dict[key] = torch.stack(
                        [torch.from_numpy(sample[key]).float() for sample in batch], dim=0)
                elif isinstance(batch[0][key], torch.Tensor):
                    ans_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)
                elif isinstance(batch[0][key], dict):
                    ans_dict[key] = _LCNuScenesTSDistillFullInternal.collate_fn(
                        [sample[key] for sample in batch])
                else:
                    ans_dict[key] = [sample[key] for sample in batch]
            return ans_dict
