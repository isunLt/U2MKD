import random
import os
import numpy as np
from copy import deepcopy
from PIL import ImageFilter
import pickle
from typing import List, Sequence, Optional
from visualize_utils import visualize_pcd



class PCDTransformTool:

    def __init__(self, pcd):
        self.pcd = deepcopy(pcd.T)

    def rotate(self, rm):
        self.pcd[:3, :] = np.dot(rm, self.pcd[:3, :])

    def translate(self, tv):
        for i in range(3):
            self.pcd[i, :] = self.pcd[i, :] + tv[i]

    def transform(self, tm):
        self.pcd = np.dot(tm, np.vstack((self.pcd[:3, :], np.ones(self.pcd.shape[1]))))[:3, :]

    def pcd2image(self, intrinsic, normalize=True):
        assert intrinsic.shape[0] == 3
        assert intrinsic.shape[1] == 3
        self.pcd = np.dot(intrinsic, self.pcd)
        if normalize:
            self.pcd = self.pcd / self.pcd[2:3, :].repeat(3, 0).reshape(3, self.pcd.shape[1])


class InstAugmentation:
    def __init__(self, instance_pkl_path, thing_list, ground_list, pair_list, add_num, num_classes, feat_dim,
                 add_order=None, class_min_num=None, class_name=None, class_weight=None, random_flip=False, random_rotate=False, random_trans=False):
        self.thing_list = thing_list
        self.ground_list = ground_list
        if class_weight is not None:
            self.instance_weight = [class_weight[i] for i in self.thing_list]
            self.instance_weight = np.sqrt(np.array(self.instance_weight))
            self.instance_weight = self.instance_weight / np.sum(self.instance_weight)

            # self.instance_weight = self.instance_weight / np.sum(self.instance_weight)
            # self.instance_weight = 1 / np.sqrt(self.instance_weight + 1e-4)
            # self.instance_weight = self.instance_weight / np.sum(self.instance_weight)
        else:
            assert len(self.thing_list) > 0
            self.instance_weight = np.array([1.0 / len(self.thing_list) for _ in thing_list])
        if class_min_num is not None:
            assert len(class_min_num) == len(thing_list)
            self.class_min_num = class_min_num
        else:
            self.class_min_num = [10] * len(thing_list)

        self.class_name = class_name
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_trans = random_trans
        self.add_num = add_num
        self.inst_root = os.path.dirname(instance_pkl_path)
        self.instance_pkl_path = instance_pkl_path
        with open(instance_pkl_path, 'rb') as f:
            self.instance_path = pickle.load(f)
        if class_name is not None:
            self.instance_path = [self.instance_path[c] for c in self.class_name]

        self.grid_size = np.array([5., 5.], dtype=np.float32)
        self.ground_classes = ground_list
        self.pair_list = pair_list
        self.num_classes = num_classes
        self.thing_class = np.zeros(shape=(num_classes,), dtype=bool)
        for c_i in thing_list:
            self.thing_class[c_i] = True
        self.feat_dim = feat_dim
        self.add_order = add_order

    def _cat_grid(self, xyz: np.ndarray):
        if isinstance(self.grid_size, list):
            self.grid_size = np.array(self.grid_size, dtype=np.float32)
        assert isinstance(self.grid_size, np.ndarray)
        grid = np.round(xyz[:, :2] / self.grid_size).astype(np.int32)
        grid -= grid.min(0, keepdims=True)
        return grid

    def ground_analyze(self, point_xyz, point_label):
        ground_info = {}
        for g_i in self.ground_list:
            g_m = point_label == g_i
            if np.sum(g_m) == 0:
                continue
            g_xyz = point_xyz[g_m]
            grid = self._cat_grid(g_xyz)
            uq, inv, count = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
            patch_center = np.zeros(shape=(uq.shape[0], g_xyz.shape[1]))
            for idx, p_id in enumerate(inv):
                patch_center[p_id] += g_xyz[idx]
            patch_center /= count.reshape(-1, 1)
            patch_center = patch_center[count >= 20]
            ground_info[g_i] = patch_center
        return ground_info

    def inst_aug(self, point_xyz, point_label, point_feat=None):

        ground_info = self.ground_analyze(point_xyz, point_label)
        instance_choice = np.random.choice(len(self.thing_list), self.add_num, replace=True, p=self.instance_weight)
        uni_inst, uni_inst_count = np.unique(instance_choice, return_counts=True)
        if self.add_order is not None:
            order = np.argsort(self.add_order[uni_inst])
            uni_inst = uni_inst[order]
            uni_inst_count = uni_inst_count[order]
        total_point_num = 0
        for inst_i, count in zip(uni_inst, uni_inst_count):
            random_choice = np.random.choice(self.instance_path[inst_i], count)
            pair_ground = self.pair_list[inst_i]
            for inst_info in random_choice:
                if isinstance(inst_info, dict):
                    path = os.path.join(self.inst_root, inst_info['path'])
                else:
                    path = os.path.join(self.inst_root, inst_info)
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :self.feat_dim]
                add_xyz = points[:, :3]
                if add_xyz.shape[0] < self.class_min_num[inst_i]:
                    continue
                center = np.mean(add_xyz, axis=0)
                min_xyz = np.min(add_xyz, axis=0)
                # max_xyz = np.max(add_xyz, axis=0)
                center[2] = min_xyz[2]
                ground_list = []
                for i in pair_ground:
                    ch = ground_info.get(i, None)
                    if ch is None or ch.shape[0] == 0:
                        continue
                    ch_i = np.random.choice(ground_info[i].shape[0], 5)
                    ground_list.append(ground_info[i][ch_i])
                ground_list = np.concatenate(ground_list, axis=0)
                ground_list = np.random.permutation(ground_list)
                break_flag = False
                for g_center in ground_list:
                    for _ in range(5):
                        if self.random_trans:
                            rand_xy = (2 * np.random.random(2) - 1) * self.grid_size / 10
                            rand_z = np.random.random(1) * 0.05
                            g_center[:2] += rand_xy
                            g_center[2] += rand_z
                        if self.random_flip:
                            long_axis = [center[0], center[1]] / (center[0] ** 2 + center[1] ** 2) ** 0.5
                            short_axis = [-long_axis[1], long_axis[0]]
                            # random flip
                            add_xyz[:, :2] = self.instance_flip(add_xyz[:, :2], [long_axis, short_axis], [center[0], center[1]])
                        if self.random_rotate:
                            rot_noise = np.random.uniform(-np.pi / 36, np.pi / 36)
                            add_xyz = self.rotate_origin(add_xyz - center, rot_noise)
                            add_xyz = add_xyz + center
                        arrow = g_center - center
                        min_xyz_a = np.min(add_xyz, axis=0) + arrow
                        max_xyz_a = np.max(add_xyz, axis=0) + arrow
                        mask_occ = point_xyz[:, 0] > min_xyz_a[0]
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 1] > min_xyz_a[1])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 2] > min_xyz_a[2])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 0] < max_xyz_a[0])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 1] < max_xyz_a[1])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 2] < max_xyz_a[2])
                        if np.sum(mask_occ) > 0:
                            occ_cls = point_label[mask_occ]
                            num_thing = np.sum(self.thing_class[occ_cls])
                            if num_thing / add_xyz.shape[0] > 0.001:
                                continue
                            elif (occ_cls.shape[0] - num_thing) / add_xyz.shape[0] > 0.05:
                                continue
                        add_label = np.ones(shape=(points.shape[0],), dtype=np.uint8) * self.thing_list[inst_i]
                        point_xyz = np.concatenate((point_xyz, add_xyz + arrow), axis=0)
                        point_label = np.concatenate((point_label, add_label), axis=0)
                        if point_feat is not None:
                            add_fea = points[:, 3:]
                            if len(point_feat.shape) == 1: point_feat = point_feat[..., np.newaxis]
                            if len(add_fea.shape) == 1: add_fea = add_fea[..., np.newaxis]
                            point_feat = np.concatenate((point_feat, add_fea), axis=0)
                        total_point_num += points.shape[0]
                        break_flag = True
                        break
                    if break_flag:
                        break
                if total_point_num > 5000:
                    break

        if point_feat is not None:
            return point_xyz, point_label, point_feat
        else:
            return point_xyz, point_label

    def instance_flip(self, points, axis, center):
        flip_type = np.random.choice(4, 1)
        points = points[:] - center
        if flip_type == 0:
            # rotate 180 degree
            points = -points + center
        elif flip_type == 1:
            # flip over long axis
            a = axis[0][0]
            b = axis[0][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center
        elif flip_type == 2:
            # flip over short axis
            a = axis[1][0]
            b = axis[1][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center

        return points

    def rotate_origin(self, xyz, radians):
        'rotate a point around the origin'
        x = xyz[:, 0]
        y = xyz[:, 1]
        new_xyz = xyz.copy()
        new_xyz[:, 0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:, 1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz


class InstAugmentationV2:
    def __init__(self, instance_pkl_path, thing_list, ground_list, pair_list, add_num, num_classes, feat_dim, feat_dim_s=5,
                 add_order=None, class_min_num=None, class_name=None, class_weight=None, random_flip=False, random_rotate=False, random_trans=False):
        self.thing_list = thing_list
        self.ground_list = ground_list
        self.instance_weight = None
        if class_weight is not None:
            self.instance_weight = [class_weight[i] for i in self.thing_list]
            self.instance_weight = np.array(self.instance_weight) / np.sum(self.instance_weight)
            # self.instance_weight = 1 / np.sqrt(self.instance_weight + 1e-4)
            # self.instance_weight = self.instance_weight / np.sum(self.instance_weight)
        else:
            assert len(self.thing_list) > 0
            # self.instance_weight = np.array([1.0 / len(self.thing_list) for _ in thing_list])
        if class_min_num is not None:
            assert len(class_min_num) == len(thing_list)
            self.class_min_num = class_min_num
        else:
            self.class_min_num = [10] * len(thing_list)

        self.class_name = class_name
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_trans = random_trans
        self.add_num = add_num
        self.inst_root = os.path.dirname(instance_pkl_path)
        self.instance_pkl_path = instance_pkl_path
        with open(instance_pkl_path, 'rb') as f:
            self.instance_path = pickle.load(f)
        if class_name is not None:
            self.instance_path = [self.instance_path[c] for c in self.class_name]

        self.grid_size = np.array([5., 5.], dtype=np.float32)
        self.ground_classes = ground_list
        self.pair_list = pair_list
        self.num_classes = num_classes
        self.thing_class = np.zeros(shape=(num_classes,), dtype=bool)
        for c_i in thing_list:
            self.thing_class[c_i] = True
        self.feat_dim_src = feat_dim_s
        self.feat_dim = feat_dim
        self.add_order = add_order

    def _cat_grid(self, xyz: np.ndarray):
        if isinstance(self.grid_size, list):
            self.grid_size = np.array(self.grid_size, dtype=np.float32)
        assert isinstance(self.grid_size, np.ndarray)
        grid = np.round(xyz[:, :2] / self.grid_size).astype(np.int32)
        grid -= grid.min(0, keepdims=True)
        return grid

    def ground_analyze(self, point_xyz, point_label):
        ground_info = {}
        for g_i in self.ground_list:
            g_m = point_label == g_i
            if np.sum(g_m) == 0:
                continue
            g_xyz = point_xyz[g_m]
            grid = self._cat_grid(g_xyz)
            uq, inv, count = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
            patch_center = np.zeros(shape=(uq.shape[0], g_xyz.shape[1]))
            for idx, p_id in enumerate(inv):
                patch_center[p_id] += g_xyz[idx]
            patch_center /= count.reshape(-1, 1)
            patch_center = patch_center[count >= 20]
            ground_info[g_i] = patch_center
        return ground_info

    def inst_aug(self, point_xyz, point_label, point_feat=None):

        ground_info = self.ground_analyze(point_xyz, point_label)
        if self.instance_weight is not None:
            instance_choice = np.random.choice(len(self.thing_list), self.add_num, replace=True, p=self.instance_weight)
        else:
            instance_choice = np.random.choice(len(self.thing_list), self.add_num, replace=True)
        uni_inst, uni_inst_count = np.unique(instance_choice, return_counts=True)
        if self.add_order is not None:
            order = np.argsort(self.add_order[uni_inst])
            uni_inst = uni_inst[order]
            uni_inst_count = uni_inst_count[order]
        total_point_num = 0
        for inst_i, count in zip(uni_inst, uni_inst_count):
            random_choice = np.random.choice(self.instance_path[inst_i], count)
            pair_ground = self.pair_list[inst_i]
            for inst_info in random_choice:
                if isinstance(inst_info, dict):
                    path = os.path.join(self.inst_root, inst_info['path'])
                else:
                    inst_info_list = inst_info.split('/')
                    path = os.path.join(self.inst_root, inst_info_list[-3], inst_info_list[-2], inst_info_list[-1])
                    # path = os.path.join(self.inst_root, inst_info)
                # points = np.fromfile(path, dtype=np.float32).reshape(-1, self.feat_dim_src)[:, :self.feat_dim] # for inst_database_train_info.pkl
                points = np.fromfile(path, dtype=np.float32).reshape(-1, self.feat_dim_src)
                # points = np.fromfile(path, dtype=np.float64).reshape(-1, 5)[:, :self.feat_dim].astype(np.float32) # for inst_database_multisweeps2_train_info.pkl
                add_xyz = points[:, :3]
                # print('add_xyz:', add_xyz.shape)
                # print('inst_i:', self.thing_list[inst_i])
                # if add_xyz.shape[0] < self.class_min_num[inst_i]:
                #     continue
                # add_label = np.ones(shape=(points.shape[0],), dtype=np.uint8) * self.thing_list[inst_i]
                # visualize_pcd(add_xyz, predict=add_label)
                center = np.mean(add_xyz, axis=0)
                min_xyz = np.min(add_xyz, axis=0)
                # max_xyz = np.max(add_xyz, axis=0)
                center[2] = min_xyz[2]
                ground_list = []
                for i in pair_ground:
                    ch = ground_info.get(i, None)
                    if ch is None or ch.shape[0] == 0:
                        continue
                    ground_list.append(ground_info[i])
                    # ch_i = np.random.choice(ground_info[i].shape[0], 5)
                    # ground_list.append(ground_info[i][ch_i])
                ground_list = np.concatenate(ground_list, axis=0)
                d_gnd = np.linalg.norm(ground_list, axis=-1)
                d_obj = np.linalg.norm(center, axis=-1)
                s_i = np.argsort(np.fabs(d_obj - d_gnd))
                ground_list = ground_list[s_i][:9, :] if ground_list.shape[0] > 9 else ground_list[s_i]
                # ground_list = np.random.permutation(ground_list)
                break_flag = False
                for g_center in ground_list:
                    for _ in range(5):
                        if self.random_trans:
                            rand_xy = (2 * np.random.random(2) - 1) * self.grid_size / 10
                            rand_z = np.random.random(1) * 0.05
                            g_center[:2] += rand_xy
                            g_center[2] += rand_z
                        if self.random_flip:
                            flip_type = np.random.choice(4, 1)
                            if flip_type == 1:
                                add_xyz[:, 0] = -add_xyz[:, 0]
                            elif flip_type == 2:
                                add_xyz[:, 1] = -add_xyz[:, 1]
                            elif flip_type == 3:
                                add_xyz[:, :2] = -add_xyz[:, :2]
                        # if self.random_flip:
                        #     long_axis = [center[0], center[1]] / (center[0] ** 2 + center[1] ** 2) ** 0.5
                        #     short_axis = [-long_axis[1], long_axis[0]]
                        #     # random flip
                        #     add_xyz[:, :2] = self.instance_flip(add_xyz[:, :2], [long_axis, short_axis], [center[0], center[1]])
                        if self.random_rotate:
                            rot_noise = np.random.uniform(-np.pi / 36, np.pi / 36)
                            add_xyz = self.rotate_origin(add_xyz - center, rot_noise)
                            add_xyz = add_xyz + center
                        rot = self.calc_rot_matrix(center, g_center)
                        add_xyz = np.dot(add_xyz, rot.T)
                        center = np.dot(center.reshape(1, 3), rot.T).flatten()
                        arrow = g_center - center
                        min_xyz_a = np.min(add_xyz, axis=0) + arrow
                        max_xyz_a = np.max(add_xyz, axis=0) + arrow
                        mask_occ = point_xyz[:, 0] > min_xyz_a[0]
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 1] > min_xyz_a[1])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 2] > min_xyz_a[2])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 0] < max_xyz_a[0])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 1] < max_xyz_a[1])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 2] < max_xyz_a[2])
                        if np.sum(mask_occ) > 0:
                            occ_cls = point_label[mask_occ]
                            num_thing = np.sum(self.thing_class[occ_cls])
                            if num_thing / add_xyz.shape[0] > 0.001:
                                continue
                            elif (occ_cls.shape[0] - num_thing) / add_xyz.shape[0] > 0.05:
                                continue
                        add_label = np.ones(shape=(points.shape[0],), dtype=np.uint8) * self.thing_list[inst_i]
                        point_xyz = np.concatenate((point_xyz, add_xyz + arrow), axis=0)
                        point_label = np.concatenate((point_label, add_label), axis=0)
                        if point_feat is not None:
                            add_fea = points[:, 3:]
                            if len(point_feat.shape) == 1: point_feat = point_feat[..., np.newaxis]
                            if len(add_fea.shape) == 1: add_fea = add_fea[..., np.newaxis]
                            point_feat = np.concatenate((point_feat, add_fea), axis=0)
                        total_point_num += points.shape[0]
                        break_flag = True
                        break
                    if break_flag:
                        break
                if total_point_num > 5000:
                    break

        if point_feat is not None:
            if len(point_feat.shape) == 1: point_feat = point_feat[..., np.newaxis]
            return point_xyz, point_label, point_feat
        else:
            return point_xyz, point_label

    def instance_flip(self, points, axis, center):
        flip_type = np.random.choice(4, 1)
        points = points[:] - center
        if flip_type == 0:
            # rotate 180 degree
            points = -points + center
        elif flip_type == 1:
            # flip over long axis
            a = axis[0][0]
            b = axis[0][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center
        elif flip_type == 2:
            # flip over short axis
            a = axis[1][0]
            b = axis[1][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center

        return points

    def rotate_origin(self, xyz, radians):
        'rotate a point around the origin'
        x = xyz[:, 0]
        y = xyz[:, 1]
        new_xyz = xyz.copy()
        new_xyz[:, 0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:, 1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz

    def calc_rot_matrix(self, obj, gnd):
        obj_ = np.array([obj[0], obj[1], 0.])
        gnd_ = np.array([gnd[0], gnd[1], 0.])
        css = np.cross(obj_, gnd_)
        cos_val = np.dot(obj, gnd) / (np.linalg.norm(obj) * np.linalg.norm(gnd))
        if  1.0 < cos_val < 1.0 + 1e-6:
            cos_val = 1.0
        elif -1.0 - 1e-6 < cos_val < -1.0:
            cos_val = -1.0
        theta = np.arccos(cos_val)
        theta = -theta if css[2] < 0 else theta  # css[2] < 0 顺时针旋转
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])  # 逆时针旋转
        return rot


class PolarMix:

    def __init__(self, instance_classes):
        self.omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
        self.instance_classes = instance_classes

    def polar_mix(self, pts1, labels1, pts2, labels2, alpha, beta):

        def swap(pt1, pt2, start_angle, end_angle, label1, label2):
            # calculate horizontal angle for each point
            yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
            yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

            # select points in sector
            idx1 = np.where((yaw1 > start_angle) & (yaw1 < end_angle))
            idx2 = np.where((yaw2 > start_angle) & (yaw2 < end_angle))

            # swap
            pt1_out = np.delete(pt1, idx1, axis=0)
            pt1_out = np.concatenate((pt1_out, pt2[idx2]))
            pt2_out = np.delete(pt2, idx2, axis=0)
            pt2_out = np.concatenate((pt2_out, pt1[idx1]))

            label1_out = np.delete(label1, idx1)
            label1_out = np.concatenate((label1_out, label2[idx2]))
            label2_out = np.delete(label2, idx2)
            label2_out = np.concatenate((label2_out, label1[idx1]))
            assert pt1_out.shape[0] == label1_out.shape[0]
            assert pt2_out.shape[0] == label2_out.shape[0]

            return pt1_out, pt2_out, label1_out, label2_out

        def rotate_copy(pts, labels, instance_classes, Omega):
            # extract instance points
            pts_inst, labels_inst = [], []
            for s_class in instance_classes:
                pt_idx = np.where((labels == s_class))
                pts_inst.append(pts[pt_idx])
                labels_inst.append(labels[pt_idx])
            pts_inst = np.concatenate(pts_inst, axis=0)
            labels_inst = np.concatenate(labels_inst, axis=0)

            # rotate-copy
            pts_copy = [pts_inst]
            labels_copy = [labels_inst]
            for omega_j in Omega:
                rot_mat = np.array([[np.cos(omega_j),
                                     np.sin(omega_j), 0],
                                    [-np.sin(omega_j),
                                     np.cos(omega_j), 0], [0, 0, 1]])
                new_pt = np.zeros_like(pts_inst)
                new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
                new_pt[:, 3] = pts_inst[:, 3]
                pts_copy.append(new_pt)
                labels_copy.append(labels_inst)
            pts_copy = np.concatenate(pts_copy, axis=0)
            labels_copy = np.concatenate(labels_copy, axis=0)
            return pts_copy, labels_copy

        pts_out, labels_out = pts1, labels1
        # swapping
        if np.random.random() < 0.5:
            pts_out, _, labels_out, _ = swap(pts1, pts2, start_angle=alpha, end_angle=beta, label1=labels1,
                                             label2=labels2)

        # rotate-pasting
        if np.random.random() < 1.0:
            # rotate-copy
            pts_copy, labels_copy = rotate_copy(pts2, labels2, self.instance_classes, self.omega)
            # paste
            pts_out = np.concatenate((pts_out, pts_copy), axis=0)
            labels_out = np.concatenate((labels_out, labels_copy), axis=0)

        return pts_out, labels_out


class LaserMix:
    """LaserMix data augmentation.

    The lasermix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Divide the point cloud into several regions according to pitch
           angles and combine the areas crossly.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        num_areas (List[int]): A list of area numbers will be divided into.
        pitch_angles (Sequence[float]): Pitch angles used to divide areas.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 prob: float = 1.0) -> None:

        self.num_areas = num_areas

        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles

        self.prob = prob

    def laser_mix_transform(self, pts1, labels1, pts2, labels2):
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = pts2
        mix_pts_semantic_mask = labels2

        points = pts1
        pts_semantic_mask = labels1

        # convert angle to radian
        pitch_angle_down = self.pitch_angles[0] / 180 * np.pi
        pitch_angle_up = self.pitch_angles[1] / 180 * np.pi

        rho = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        pitch = np.arctan2(points[:, 2], rho)
        pitch = np.clip(pitch, pitch_angle_down + 1e-5, pitch_angle_up - 1e-5)

        mix_rho = np.sqrt(mix_points[:, 0]**2 + mix_points[:, 1]**2)
        mix_pitch = np.arctan2(mix_points[:, 2], mix_rho)
        mix_pitch = np.clip(mix_pitch, pitch_angle_down + 1e-5, pitch_angle_up - 1e-5)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(pitch_angle_up, pitch_angle_down, num_areas + 1)

        out_points = []
        out_pts_semantic_mask = []
        mix_mask = []
        for i in range(num_areas):
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]
            if i % 2 == 0:  # pick from original point cloud
                idx = (pitch > start_angle) & (pitch <= end_angle)
                out_points.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx])
                mix_mask.append(np.ones_like(pts_semantic_mask[idx]))
            else:  # pickle from mixed point cloud
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                out_points.append(mix_points[idx])
                out_pts_semantic_mask.append(mix_pts_semantic_mask[idx])
                mix_mask.append(np.zeros_like(mix_pts_semantic_mask[idx]))
        out_points = np.concatenate(out_points, axis=0)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        mix_mask = np.concatenate(mix_mask, axis=0).astype(bool)

        return out_points, out_pts_semantic_mask, mix_mask

    def transform(self, pts1, labels1, pts2, labels2):
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return pts1, labels1

        pts1, labels1, mix_mask = self.laser_mix_transform(pts1, labels1, pts2, labels2)

        return pts1, labels1, mix_mask

