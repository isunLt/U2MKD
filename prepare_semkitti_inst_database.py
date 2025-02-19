import os
import numpy as np
import mlcrate as mlc
from functools import partial
import pickle

from nuscenes import NuScenes as NuScenes_devkit
from visualize_utils import SemKITTI_label_name_19, visualize_pcd, MapSemKITTI2NUSC

from tqdm import tqdm
import yaml

DATA_ROOT = '/data2/stf/datasets/semantickitti'
SPLIT = 'train'
DATABASE_SAVE_DIR = os.path.join(DATA_ROOT, 'inst_database_' + SPLIT)
INST_DBINFO_PKL_SAVE_PATH = os.path.join(DATA_ROOT, 'inst_database_train_info.pkl')

config_path = os.path.join(DATA_ROOT, 'semantic-kitti.yaml')
with open(config_path, 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)

learning_map = semkittiyaml['learning_map']

THING_LIST = [1, 2, 3, 4, 5, 6, 7, 8]
MIN_INST_POINT = 40
INST_DBINFO_PKL = dict()
for c_i in THING_LIST:
    INST_DBINFO_PKL[SemKITTI_label_name_19[c_i]] = []

def prepare_file_list():
    seqs = []
    if SPLIT == 'train':
        seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    elif SPLIT == 'val':
        seqs = ['08']
    elif SPLIT == 'test':
        seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    pcd_info_list = []
    for seq in seqs:
        for pcd_name in sorted(os.listdir(os.path.join(DATA_ROOT, 'sequences', seq, 'velodyne'))):
            info = {}
            lidar_token = '%s_%s' % (seq, pcd_name[:-4])
            lidar_path = os.path.join(DATA_ROOT, 'sequences', seq, 'velodyne', str(pcd_name))
            lidar_label_path = lidar_path.replace('velodyne', 'labels')[:-3] + 'label'
            info['lidar_token'] = lidar_token
            info['path'] = lidar_path
            info['label_path'] = lidar_label_path
            pcd_info_list.append(info)
    return pcd_info_list

def process_one_sequences(info: dict, save_file=True):

    global INST_DBINFO_PKL

    lidar_token = info['lidar_token']
    lidar_path = info['path']
    label_path = info['label_path']
    point_xyzi = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    anno = np.fromfile(label_path, dtype=np.int32).reshape([-1, 1])
    sem_label = anno & 0xFFFF
    sem_label = np.vectorize(learning_map.__getitem__)(sem_label).flatten()
    panoptic_label = anno >> 16
    if not save_file:
        visualize_pcd(point_xyzi[:, :3], predict=np.vectorize(MapSemKITTI2NUSC.__getitem__)(sem_label.reshape([-1, 1])), target=panoptic_label)

    for thing_id in THING_LIST:
        thing_mask = np.zeros_like(sem_label, dtype=bool)  # 每个类别一个mask
        thing_mask[sem_label == thing_id] = True  # 所有属于thing_id的类
        panoptic_label_thing = panoptic_label[thing_mask]  # 全景标签存在小于 2^16 的异常点
        unique_inst_label = np.unique(panoptic_label_thing)

        thing_name = SemKITTI_label_name_19[thing_id]
        for uq_inst_label in unique_inst_label:  # 对于每一个实例
            index = np.where(panoptic_label == uq_inst_label)[0]
            if index.shape[0] < MIN_INST_POINT:  # 如果
                continue
            if np.sum(panoptic_label[index]) == 0:
                continue
            if save_file:
                dir_path = os.path.join(DATABASE_SAVE_DIR, SemKITTI_label_name_19[thing_id])
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path)
                    except OSError:
                        print(f'Error occurred when creating ground truth mask dir "{dir_path}".')
                    else:
                        print(f'dir created "{dir_path}.')
                file_path = os.path.join(dir_path, '%s_%s_%s.bin' % (str(lidar_token), str(thing_name), str(uq_inst_label)))
                inst_points = point_xyzi[index, :]
                if not os.path.exists(file_path):
                    inst_points.tofile(file_path)
                    INST_DBINFO_PKL[thing_name].append(file_path)
            else:
                inst_points = point_xyzi[index, :]
                inst_sem_label = sem_label[index]
                inst_pano_label = panoptic_label[index]
                visualize_pcd(inst_points[:, :3], predict=np.vectorize(MapSemKITTI2NUSC.__getitem__)(inst_sem_label.reshape([-1, 1])), target=inst_pano_label)

# def main_eval_saved_file():
#
#     def _load_image(filepath: str):
#         return np.array(Image.open(filepath).convert('RGB'))
#
#     def _load_superpixel(name_list: list[str]):
#         sp_list = []
#         for name in name_list:
#             token_list = name.split('/')
#             seq, im_name = token_list[-3], token_list[-1]
#             sp_name = os.path.join(SAVE_DIR, SPLIT, str(seq), 'seeds', im_name[:-5] + '.npy')
#             sp_list.append(np.load(sp_name))
#         return sp_list
#
#     keyframe_path = os.path.join(DATA_ROOT, SPLIT, 'keyframes.txt')
#     with open(keyframe_path, 'r') as f:
#         keyframe_list = f.read().splitlines()
#     for path in keyframe_list:
#         im_list, name_list = _load_image(seq_path=path)
#         sp_list = _load_superpixel(name_list)
#         for im, sp in zip(im_list, sp_list):
#             visualize_img(im, superpixel=sp)
#         # process_one_sequences(path, save_file=True)


def main_save_npy_multiprocess():
    name_token_pair = prepare_file_list()
    pool = mlc.SuperPool(24)
    pool.map(partial(process_one_sequences, save_file=True), name_token_pair, description='process nusc instdb %s' % str(SPLIT))
    for k, v in INST_DBINFO_PKL.items():
        print(f'load {len(v)} {k} database infos')
    with open(INST_DBINFO_PKL_SAVE_PATH, 'wb') as f:
        pickle.dump(INST_DBINFO_PKL, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('pkl saved at', INST_DBINFO_PKL_SAVE_PATH)


if __name__ == '__main__':
    name_token_pair = prepare_file_list()
    for pair in tqdm(name_token_pair):
        process_one_sequences(pair, save_file=True)
    for k, v in INST_DBINFO_PKL.items():
        print(f'load {len(v)} {k} database infos')
    with open(INST_DBINFO_PKL_SAVE_PATH, 'wb') as f:
        pickle.dump(INST_DBINFO_PKL, f)
    print('pkl saved at', INST_DBINFO_PKL_SAVE_PATH)

# if __name__ == '__main__':
#     main_eval_saved_file()

# if __name__ == '__main__':
#     main_save_npy_multiprocess()

