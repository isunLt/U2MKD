import itertools
import os
import numpy as np

import torch
from typing import List

import open3d as o3d
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


CAM_CHANNELS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

VIEW_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

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

IDX2COLOR_16 = [(0, 0, 0),
                (112, 128, 144),  # barrier 蓝灰色
                (220, 20, 60),  # bicycle 玫红色
                (255, 127, 80),  # bus
                (255, 158, 0),  # car 黄色
                (233, 150, 70),  # construction_vehicle 工程车 浅一点的橙色
                (255, 61, 99),  # motorcycle 桃红色
                (0, 0, 230),  # pedestrian 蓝色
                (47, 79, 79),  # traffic_cone 锥形交通路标 灰绿色
                (255, 140, 0),  # trailer 拖车 橙色
                (255, 99, 71),  # truck 卡车
                (0, 207, 191),  # driveable_surface 蓝绿色
                (175, 0, 75),  # other_flat 紫红色
                (75, 0, 75),  # sidewalk 紫色
                (112, 180, 60),  # terrain 草绿色
                (222, 184, 135),  # manmade 土黄色
                (0, 175, 0)]  # vegetation 深绿色

IDX2COLOR_22 = [(0, 0, 0),
                (112, 128, 144),  # barrier 蓝灰色 1
                (220, 20, 60),  # bicycle 玫红色 2
                (255, 127, 80),  # bus 3
                (255, 158, 0),  # car 黄色 4
                (233, 150, 70),  # construction_vehicle 工程车 浅一点的橙色 5
                (255, 61, 99),  # motorcycle 桃红色 6
                (0, 0, 230),  # pedestrian 蓝色 7
                (47, 79, 79),  # traffic_cone 锥形交通路标 灰绿色 8
                (255, 140, 0),  # trailer 拖车 橙色 9
                (255, 99, 71),  # truck 卡车 10
                (0, 207, 191),  # driveable_surface 蓝绿色 11
                (175, 0, 75),  # other_flat 紫红色 12
                (75, 0, 75),  # sidewalk 紫色 13
                (112, 180, 60),  # terrain 草绿色 14
                (222, 184, 135),  # manmade 土黄色 15
                (0, 175, 0),  # vegetation 深绿色 16
                (255, 40, 200),  # bicyclist 粉红色 17
                (150, 30, 90),  # motorcyclist 紫红色 18
                (150, 255, 170),  # lane_marker 青绿色 19
                (255, 0, 0),  # traffic_sign 大红色 20
                (255, 150, 150),  # curb 粉红色 21
                (255, 240, 150),  # pole 淡黄色 22
                ]

SemKITTI_label_name_16 = {
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

SemKITTI_label_name_19 = {
    0: 'noise',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}

SemKITTI_label_name_22 = {
    0: 'noise',  #
    1: 'car',  #
    2: 'truck',  #
    3: 'bus',  #
    4: 'other_vehicle',  #
    5: 'motorcyclist',  #
    6: 'bicyclist',  #
    7: 'pedestrian',  #
    8: 'sign',  #
    9: 'traffic_light',  #
    10: 'pole',  #
    11: 'construction_cone',  #
    12: 'bicycle',  #
    13: 'motorcycle',  #
    14: 'building',  #
    15: 'vegetation',  #
    16: 'tree_trunk',  #
    17: 'curb',  #  路沿
    18: 'road',  #
    19: 'lane_marker',  #
    20: 'other_ground',  #
    21: 'walkable',  #
    22: 'sidewalk'  #
}

MapSemKITTI2NUSC = {
    0: 0,
    1: 4,
    2: 2,
    3: 6,
    4: 10,
    5: 5,
    6: 7,
    7: 17,
    8: 18,
    9: 11,
    10: 12,
    11: 13,
    12: 12,
    13: 15,
    14: 1,
    15: 16,
    16: 16,
    17: 14,
    18: 22,
    19: 20,
}

MapWaymo2NUSC = {
    0: 0,  # noise
    1: 4,  # car
    2: 10,  # truck
    3: 3,  # bus
    4: 5,  # other-vehicle
    5: 18,  # motorcyclist
    6: 17,  # bicyclist
    7: 7,  # pedestrian
    8: 8,  # sign
    9: 20,  # traffic_light
    10: 22,  # pole
    11: 1,  # construction_cone
    12: 2,  # bicycle
    13: 6,  # motorcycle
    14: 15,  # building
    15: 16,  # vegetation
    16: 9,  # tree_trunk
    17: 21,  # curb 路沿
    18: 11,  # road
    19: 19,  # lane_marker
    20: 12,  # other_ground
    21: 14,  # walkable
    22: 13  # sidewalk
}

def load_bin_file(bin_path: str) -> np.ndarray:
    """
    Loads a .bin file containing the labels.
    :param bin_path: Path to the .bin file.
    :return: An array containing the labels.
    """
    assert os.path.exists(bin_path), 'Error: Unable to find {}.'.format(bin_path)
    bin_content = np.fromfile(bin_path, dtype=np.uint8)
    assert len(bin_content) > 0, 'Error: {} is empty.'.format(bin_path)

    return bin_content


def visualize_pcd(xyz, **kwargs):
    """
    使用open3d渲染点云
    Args:
        xyz: <ndarray> [N, 3] 点云三维坐标xyz
        **kwargs: 可选参数
        1. predict <ndarray> [N,] 网络预测的点云标签, 第二维取值范围[0, num_class];
        2. target <ndarray> [N,] 点云标签真值
        3. view <ndarray> [N,] 每个点所在相机视野标签, 第二维取值范围[0,6)
        4. rgb <ndarray> [N, 3] 每个点的颜色, 取值范围[0, 255]
        5. select_inds <ndarray> bool标签[N, ]或者序号标签[npoint, ]
    Returns:

    """

    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if k == "predict":
            predict_color = o3d.utility.Vector3dVector(np.array([IDX2COLOR_22[int(c % 23)] for c in v]) / 255.0)
            print("load predict, render with W")
        elif k == "target":
            gt_color = o3d.utility.Vector3dVector(np.array([IDX2COLOR_22[int(c % 23)] for c in v]) / 255.0)
            print("load target, render with Q")
        elif k == "view":
            view_color = o3d.utility.Vector3dVector(
                np.array([VIEW_COLORS[c] if c != -1 else (255, 255, 255) for c in v]) / 255.0)
        elif k == 'rgb':
            rgb_color = o3d.utility.Vector3dVector(v / 255.0)
        elif k == 'select_inds':
            s_color = np.ones((xyz.shape[0], 3), dtype=np.float32) / 2
            s_color[v, :] = np.array([1., 0., 0.])
            s_color = o3d.utility.Vector3dVector(s_color)

    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])

    def render_gt_color_callback(viewer):
        if "target" in kwargs.keys():
            pcd.colors = gt_color
            viewer.update_geometry(pcd)
            print("render target")
        else:
            print("No ground truth color provided")

    def render_predict_color_callback(viewer):
        if "predict" in kwargs.keys():
            pcd.colors = predict_color
            viewer.update_geometry(pcd)
            print("render predict")
        else:
            print("No predict color provided")

    def render_view_color_callback(viewer):
        if "view" in kwargs.keys():
            pcd.colors = view_color
            viewer.update_geometry(pcd)
        else:
            print("No view color provided")

    def render_rgb_color_callback(viewer):
        if 'rgb' in kwargs.keys():
            pcd.colors = rgb_color
            viewer.update_geometry(pcd)
        else:
            print("No RGB color provided")

    def render_select_points_callback(viewer):
        if 'select_inds' in kwargs.keys():
            pcd.colors = s_color
            viewer.update_geometry(pcd)
        else:
            print("No select inds provided")

    def save_viewpoint_callback(viewer):
        param = viewer.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.cuda.pybind.io.write_pinhole_camera_parameters('viewpoint_param.json', param)
        print('save viewpoint param')

    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window()
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([1., 1., 1.])
    viewer.register_key_callback(ord("Q"), render_gt_color_callback)
    viewer.register_key_callback(ord("W"), render_predict_color_callback)
    viewer.register_key_callback(ord("V"), render_view_color_callback)
    viewer.register_key_callback(ord("R"), render_rgb_color_callback)
    viewer.register_key_callback(ord("S"), render_select_points_callback)
    viewer.register_key_callback(ord("P"), save_viewpoint_callback)
    viewer.add_geometry(pcd)
    if kwargs.get('viewpoint', None) is not None:
        path = kwargs.get('viewpoint', None)
        if os.path.exists(path):
            ctr = viewer.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters(kwargs.get('viewpoint'))
            ctr.convert_from_pinhole_camera_parameters(param)
        else:
            print('view point file not exist!')
    viewer.run()
    viewer.destroy_window()


def visualize_img(image: np.ndarray, **kwargs):
    """
    使用Image可视化图像
    :param image: <np.ndarray, [H, W, 3]>
    :param kwargs:
    1. predict <np.ndarray, [H, W]> 标签
    2. points <np.ndarray, [N, 3]> N个点, 0,1 -> w,h; 2->label
    :return:
    """

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().astype(np.uint8)

    oh, ow, c = image.shape
    assert image.ndim == 3
    image = Image.fromarray(image).convert(mode='RGB')

    if len(kwargs) == 0:
        plt.imshow(image)
        plt.show()
    else:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if k == 'predict':
                h, w = v.shape
                image_resize = image
                if h != image.height or w != image.width:
                    from torchvision.transforms import transforms
                    trans = transforms.Resize(size=v.shape)
                    image_resize = trans(image)
                color = np.array([IDX2COLOR_22[c] for c in v.flatten()]).reshape((h, w, 3)).astype(np.uint8)
                color = Image.fromarray(color).convert(mode='RGB')
                mix = Image.blend(image_resize, color, alpha=0.25)
                plt.imshow(mix)
            elif k == 'point':
                co, l = v[:, :2], v[:, 2]
                co[:, 0] = (co[:, 0] + 1.0) / 2 * (ow - 1.0)
                co[:, 1] = (co[:, 1] + 1.0) / 2 * (oh - 1.0)
                co = np.floor(co).astype(np.int32)
                l = l.astype(np.int32)
                color = [IDX2COLOR_22[c % 23] for c in l.flatten()]
                imagedraw = ImageDraw.Draw(image)
                rad = 0.01
                for (x, y), c in zip(co, color):
                    imagedraw.ellipse(xy=[x - rad, y - rad, x + rad, y + rad], fill=c)
                plt.imshow(image)
            elif k == 'select_inds':
                co, l = v[:, :2], v[:, 2]
                co[:, 0] = (co[:, 0] + 1.0) / 2 * (ow - 1.0)
                co[:, 1] = (co[:, 1] + 1.0) / 2 * (oh - 1.0)
                co = np.floor(co).astype(np.int32)
                l = l.astype(bool)
                color = np.ones(shape=[co.shape[0], 3], dtype=np.float32) / 2
                color[l] = np.array([1., 0., 0.], dtype=np.float32)
                color *= 255
                color = color.astype(np.uint8)
                imagedraw = ImageDraw.Draw(image)
                rad = 0.25
                for (x, y), c in zip(co, color):
                    imagedraw.ellipse(xy=[x - rad, y - rad, x + rad, y + rad], fill=tuple(c))
                plt.imshow(image)
            elif k == 'superpixel':
                h, w = v.shape
                v = v.astype(np.int32)
                # color = np.array([IDX2COLOR_16[1:][c % 16] for c in v.flatten()]).reshape((h, w, 3)).astype(np.uint8)
                color = np.array([IDX2COLOR_22[c % 23] for c in v.flatten()]).reshape((h, w, 3)).astype(np.uint8)
                color = Image.fromarray(color).convert(mode='RGB')
                mix = Image.blend(image, color, alpha=0.25)
                plt.imshow(mix)
            elif k == 'heatmap':
                from matplotlib.cm import get_cmap
                h, w = v.shape
                color = get_cmap('bwr')(v)[:, :, :3] * 255
                color = Image.fromarray(color.astype(np.uint8)).convert(mode='RGB')
                mix = Image.blend(image, color, alpha=0.25)
                plt.imshow(mix)
            plt.show()