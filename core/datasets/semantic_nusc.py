import os
import numpy as np

import torch
from torch.utils import data

from torchpack.utils.config import configs

from nuscenes import NuScenes as NuScenes_devkit
from torchvision.transforms import transforms

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate

from core.datasets.utils import InstAugmentation, PCDTransformTool, PolarMix, LaserMix, InstAugmentationV2
from nuscenes.utils.data_classes import transform_matrix
from pyquaternion import Quaternion
from functools import reduce


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


class NuScenes(dict):
    def __init__(self, root, voxel_size, version, verbose, **kwargs):
        self.nusc = NuScenes_devkit(dataroot=root, version=version, verbose=verbose)
        super(NuScenes, self).__init__({
            "train": _NuScenesInternal(nusc=self.nusc, voxel_size=voxel_size, split="train"),
            "val": _NuScenesInternal(nusc=self.nusc, voxel_size=voxel_size, split="val")
        })


class _NuScenesInternal(data.Dataset):
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

    CLASS_DISTRIBUTE = [2.87599770e-01, 7.84806427e-03, 1.19217528e-04, 3.88372281e-03, 3.21376629e-02, 1.27727921e-03,
                        3.60467902e-04, 1.95227505e-03, 6.20954881e-04, 4.13906749e-03, 1.33608580e-02, 2.67327832e-01,
                        7.21896959e-03, 5.92055787e-02, 5.92833998e-02, 1.50278018e-01, 1.03386862e-01]

    INST_DATABASE_DISTRIBUTE = [0, 60272, 2561, 8091, 156414, 6908, 3036, 35011, 13188, 14186, 41250]
    # INST_DATABASE_DISTRIBUTE = [0, 54382, 2580, 8117, 156743, 7572, 2952, 34257, 13303, 13862, 41658] # multisweeps

    CLASS_NAME = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                  'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',  'truck',]

    def __init__(self, nusc, voxel_size, split="train", **kwargs):
        self.nusc = nusc
        self.voxel_size = voxel_size
        self.split = split
        self.ignored_labels = np.sort([0])
        self.augment = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

        self.num_classes = configs['data']['num_classes']
        self.flip_aug = configs['dataset']['flip_aug']
        self.rotate_aug = configs['dataset']['rotate_aug']
        self.translate_std = configs.dataset.get('translate_std', None)
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
        return len(self.sample)

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

    def load_single_file(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        # raw_num = pts.shape[0]

        if self.split == "test":
            labels_ = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            lidar_label_path = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"])
            labels_ = np.fromfile(lidar_label_path, dtype=np.uint8)
            labels_ = np.vectorize(self.labels_mapping.__getitem__)(labels_).flatten()

        return pts, labels_

    def __getitem__(self, index):
        sample = self.sample[index]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_channel["filename"])
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])[:, :4]  # N, 4
        raw_num = pts.shape[0]

        if self.split == "test":
            labels_ = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            lidar_label_path = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"])
            labels_ = np.fromfile(lidar_label_path, dtype=np.uint8)
            labels_ = np.vectorize(self.labels_mapping.__getitem__)(labels_).flatten()

        if 'train' in self.split and self.inst_aug:
            pts, labels_, intensity = self.inst_augmenter.inst_aug(pts[:, :3], labels_, pts[:, 3:])
            pts = np.concatenate([pts, intensity], axis=-1)

        if self.multisweeps != 0:
            agg_pts, agg_ts = self._aggregate_lidar_sweeps(sample_ref=sample, nsweeps=self.multisweeps, only_past=self.only_past)
            # print('agg_pts:', len(agg_pts))
            agg_pts = np.concatenate([pts] + agg_pts, axis=0)
            agg_ts = np.concatenate([np.zeros(shape=(pts.shape[0],))] + agg_ts, axis=0)
            keyframe_mask = (agg_ts == 0)
            num = np.sum(~keyframe_mask)
            labels_ = np.concatenate([labels_, np.full(shape=(int(num),), fill_value=self.ignored_labels, dtype=np.uint8)], axis=0)
            pts = agg_pts

        if 'train' in self.split and self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                pts[:, 0] = -pts[:, 0]
            elif flip_type == 2:
                pts[:, 1] = -pts[:, 1]
            elif flip_type == 3:
                pts[:, :2] = -pts[:, :2]

        pts_cp = np.zeros_like(pts)
        if 'train' in self.split and self.rotate_aug:
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

        if 'train' in self.split and self.translate_std:
            noise_translate = np.array([np.random.normal(0, self.translate_std[0], 1),
                                        np.random.normal(0, self.translate_std[1], 1),
                                        np.random.normal(0, self.translate_std[2], 1)]).T

            pts_cp[:, :3] += noise_translate

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
            'lidar_token': lidar_token,
            'num_vox': voxel_full.shape[0],
        }

        if self.multisweeps != 0:
            feed_dict['keyframe_mask'] = SparseTensor(keyframe_mask[inds], voxel_full)
            feed_dict['keyframe_mask_full'] = SparseTensor(keyframe_mask, voxel)

        return feed_dict

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
                    ans_dict[key] = sparse_collate_fn(
                        [sample[key] for sample in batch])
                else:
                    ans_dict[key] = [sample[key] for sample in batch]
            return ans_dict