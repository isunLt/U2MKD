import os
import os.path

import yaml
import numpy as np
from PIL import Image

from torchvision.transforms import transforms
from torchpack.utils.config import configs

import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from core.datasets.utils import InstAugmentationV2
from copy import deepcopy

__all__ = ['SemanticKITTI']

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class SemanticKITTI(dict):

    def __init__(self, root, voxel_size, **kwargs):
        config_path = os.path.join(root, 'semantic-kitti.yaml')
        with open(config_path, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        # learning_map = semkittiyaml['learning_map']
        root = os.path.join(root, 'sequences')
        super().__init__({
            'train': _SemanticKITTIInternal(root, voxel_size, split='train', yaml_files=semkittiyaml),
            'val': _SemanticKITTIInternal(root, voxel_size, split='val', yaml_files=semkittiyaml)
        })


class _SemanticKITTIInternal:

    CLASS_NAME = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
    INST_DATABASE_DISTRIBUTE = [0, 115689, 2979, 2315, 2402, 5988, 4545, 916, 479]

    def __init__(self,
                 root,
                 voxel_size,
                 split,
                 yaml_files):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.yaml_files = yaml_files
        self.labels_mapping = yaml_files['learning_map']
        self.seqs = []
        if split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.num_points = 80000

        self.pcd_files = []
        for seq in self.seqs:
            seq_list = []
            for pcd_name in sorted(os.listdir(os.path.join(self.root, seq, 'velodyne'))):
                seq_list.append(os.path.join(self.root, seq, 'velodyne', str(pcd_name)))
            self.pcd_files.append(seq_list)

        self.P_list = []
        self.Tr_list = []
        for seq in self.seqs:
            with open(os.path.join(self.root, seq, 'calib.txt'), 'r') as calib:
                P = []
                for idx in range(4):
                    line = calib.readline().rstrip('\n')[4:]
                    data = line.split(' ')
                    P.append(np.array(data, dtype=np.float32).reshape(3, -1))
                self.P_list.append(P[2])
                line = calib.readline().rstrip('\n')[4:]
                data = line.split(' ')
                self.Tr_list.append(np.array(data, dtype=np.float32).reshape((3, 4)))

        self.pose_list = []
        for s_i, seq in enumerate(self.seqs):
            pl = []
            with open(os.path.join(self.root, seq, 'poses.txt'), 'r') as pose_file:
                for _ in self.pcd_files[s_i]:
                    p = pose_file.readline().rstrip('\n')
                    p = p.split(' ')
                    pl.append(np.array(p, dtype=np.float32).reshape((3, 4)))
                self.pose_list.append(pl)

        self.id_pair = []
        for s_i, seq in enumerate(self.pcd_files):
            for p_i, _ in enumerate(seq):
                self.id_pair.append((s_i, p_i))

        self.num_classes = configs['data']['num_classes']
        self.rotate_aug = configs['dataset']['rotate_aug']
        self.flip_aug = configs['dataset']['flip_aug']
        self.inst_aug = configs['dataset']['inst_aug']
        self.min_coords = np.array([-50, -50, -4], dtype=np.float32)
        self.max_coords = np.array([50, 50, 4], dtype=np.float32)
        self.inst_pkl_path = configs['dataset']['inst_pkl_path']
        self.translate_std = configs.dataset.get('translate_std', None)
        # self.cls_weight = self.fetch_class_weight()
        # self.cls_weight = [3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704,
        #                    10.1922, 1.6155, 4.2187, 1.9385, 5.5455, 2.0198, 2.6261, 1.3212,
        #                    5.1102, 2.5492, 5.8585, 7.3929]
        if self.inst_aug:
            inst_pkl_path = configs['dataset']['inst_pkl_path']
            thing_list = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
            ground_list = np.array([9, 10, 11, 12, 17], dtype=np.uint8)
            pair_list = [[9, 10], [9, 10, 11], [9, 10, 11], [9, 10], [9, 10], [9, 10, 11, 12, 17], [9, 10, 11, 12],
                         [9, 10, 11, 12]]
            inst_add_num = configs['dataset']['inst_add_num']
            self.inst_augmenter = InstAugmentationV2(instance_pkl_path=inst_pkl_path, thing_list=thing_list,
                                                     ground_list=ground_list, pair_list=pair_list, add_num=inst_add_num,
                                                     num_classes=self.num_classes, class_name=self.CLASS_NAME,
                                                     class_weight=self.INST_DATABASE_DISTRIBUTE,
                                                     class_min_num=[300, 50, 50, 300, 300, 25, 25, 25],
                                                     random_flip=False, random_rotate=False, random_trans=True,
                                                     feat_dim=4, feat_dim_s=4)
        self.multisweeps = configs['dataset']['multisweeps']['num_sweeps']
        self.only_past = configs['dataset']['multisweeps']['only_past']


    def __len__(self):
        return len(self.id_pair)

    def _load_single_frame(self, s_i, p_i):
        filepath = self.pcd_files[s_i][p_i]
        pts = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))
        if self.split == 'test':
            labels_ = np.expand_dims(np.zeros_like(pts[:, 0], dtype=int), axis=1)
        else:
            lidar_label_path = filepath.replace('velodyne', 'labels')[:-3] + 'label'
            anno = np.fromfile(lidar_label_path, dtype=np.int32).reshape([-1, 1])
            sem_labels = anno & 0xFFFF
            labels_ = np.vectorize(self.labels_mapping.__getitem__)(sem_labels).flatten()
        return pts, labels_, anno

    def _inv_trans_matrix(self, m: np.ndarray):
        r, t = m[:3, :3], m[:3, 3]
        inv = np.zeros(shape=[4, 4], dtype=m.dtype)
        inv[:3, :3] = r.T
        inv[:3, 3] = - r.T @ t
        inv[3, 3] = 1
        return inv

    def _aggregate_lidar_sweeps(self, s_i, p_i, nsweeps=5):
        ref_tv = self.Tr_list[s_i]
        ref_pose = self.pose_list[s_i][p_i]
        ref_tv_inv = self._inv_trans_matrix(ref_tv)
        ref_pose_inv = self._inv_trans_matrix(ref_pose)
        start_idx = p_i - nsweeps if (p_i - nsweeps) > 0 else 0
        end_idx = p_i + nsweeps if (p_i + nsweeps) < len(self.pcd_files[s_i]) else p_i
        agg_pts, agg_labels = [], []
        for i in range(start_idx, end_idx):
            if i == p_i:
                continue
            pts_i, l_i, _ = self._load_single_frame(s_i, i)
            pts_i_homo = np.column_stack((pts_i[:, :3], np.array([1] * pts_i.shape[0], dtype=pts_i.dtype)))
            tv = self.Tr_list[s_i]
            tv_homo = np.row_stack((tv, np.array([0, 0, 0, 1], dtype=tv.dtype)))
            pts_i_homo = np.matmul(tv_homo, pts_i_homo.T)
            pose = self.pose_list[s_i][i]
            pose_homo = np.row_stack((pose, np.array([0,0,0,1], dtype=pose.dtype)))
            pts_i_homo = np.matmul(pose_homo, pts_i_homo)
            pts_i_homo = np.matmul(ref_pose_inv, pts_i_homo)
            pts_i_homo = np.matmul(ref_tv_inv, pts_i_homo).T
            pts_i_homo[:, 3] = pts_i[:, 3]
            # agg_pts.append(pts_i_homo[:, :3])
            agg_pts.append(pts_i_homo)
            agg_labels.append(l_i)
        return agg_pts, agg_labels

    def __getitem__(self, index):
        s_i, p_i = self.id_pair[index]
        pts, labels_, inst_label = self._load_single_frame(s_i, p_i)
        pts_num_raw = pts.shape[0]

        if 'train' in self.split and self.inst_aug:
            pts, labels_, intensity = self.inst_augmenter.inst_aug(point_xyz=pts[:, :3],
                                                                   point_label=labels_,
                                                                   point_feat=pts[:, 3])
            pts = np.concatenate([pts, intensity], axis=-1)
            labels_ = labels_.flatten()
            raw_mask = np.ones_like(labels_).astype(bool)
            raw_mask[:pts_num_raw] = False

        if self.multisweeps != 0:
            agg_pts, agg_labels = self._aggregate_lidar_sweeps(s_i, p_i, nsweeps=self.multisweeps)
            agg_pts = np.concatenate([pts] + agg_pts, axis=0)
            agg_labels = np.concatenate([labels_] + agg_labels, axis=0).reshape((-1, 1))
            # keyframe_mask = np.arange(agg_pts.shape[0]) < pts.shape[0]
            labels_ = agg_labels.flatten()
            pts = agg_pts

        # if 'train' in self.split and self.min_coords is not None and self.max_coords is not None:
        #     mask = self._remove_far_points(pts)
        #     pts = pts[mask]
        #     labels_ = labels_[mask]

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

        # if 'train' in self.split and self.translate_std:
        #     noise_translate = np.array([np.random.normal(0, self.translate_std[0], 1),
        #                                 np.random.normal(0, self.translate_std[1], 1),
        #                                 np.random.normal(0, self.translate_std[2], 1)]).T
        #
        #     pts_cp[:, :3] += noise_translate

        pts_cp[:, 3] = pts[:, 3]  # block为经过随机旋转和放缩的点云 [N, 4] -> x,y,z,sig
        voxel = np.round(pts_cp[:, :3] / self.voxel_size).astype(np.int32)  # voxelization
        voxel -= voxel.min(0, keepdims=1)  # 将体素坐标最小值为0

        feat_ = pts_cp.astype(np.float32)

        _, inds, inverse_map = sparse_quantize(voxel,
                                               return_index=True,
                                               return_inverse=True)
        # if 'train' in self.split:
        #     if len(inds) > self.num_points:
        #         inds = np.random.choice(inds, self.num_points, replace=False)

        voxel_full = voxel[inds]
        feat_full = feat_[inds]
        labels_full = labels_[inds]
        lidar = SparseTensor(feat_full, voxel_full)
        labels = SparseTensor(labels_full, voxel_full)
        labels_ = SparseTensor(labels_, voxel)
        inverse_map = SparseTensor(inverse_map, voxel)

        feed_dict =  {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'num_vox': voxel_full.shape[0],
            # 'raw_mask': SparseTensor(raw_mask[inds], voxel_full)
        }

        # if self.multisweeps != 0:
        #     feed_dict['keyframe_mask'] = SparseTensor(keyframe_mask[inds], voxel_full)
        #     feed_dict['keyframe_mask_full'] = SparseTensor(keyframe_mask, voxel)

        return feed_dict

    @staticmethod
    def collate_fn(batch):
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