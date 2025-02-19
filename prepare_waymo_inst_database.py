import os
import numpy as np
import mlcrate as mlc
from functools import partial
import pickle

from visualize_utils import SemKITTI_label_name_22, visualize_pcd, MapWaymo2NUSC

from tqdm import tqdm

DATA_ROOT = '/data1/stf/waymo_bin_full_v1p43'
SAVE_DB_DIR = '/data1/stf/waymo_bin_full_v1p43'
SPLIT = 'training'
DATA_ROOT = os.path.join(DATA_ROOT, SPLIT)
DATABASE_SAVE_DIR = os.path.join(SAVE_DB_DIR, 'inst_database_' + SPLIT)
INST_DBINFO_PKL_SAVE_PATH = os.path.join(SAVE_DB_DIR, 'inst_database_%s_info.pkl' % SPLIT)

# labels_mapping = {
#     1: 0,
#     5: 0,
#     7: 0,
#     8: 0,
#     10: 0,
#     11: 0,
#     13: 0,
#     19: 0,
#     20: 0,
#     0: 0,
#     29: 0,
#     31: 0,
#     9: 1,
#     14: 2,
#     15: 3,
#     16: 3,
#     17: 4,
#     18: 5,
#     21: 6,
#     2: 7,
#     3: 7,
#     4: 7,
#     6: 7,
#     12: 8,
#     22: 9,
#     23: 10,
#     24: 11,
#     25: 12,
#     26: 13,
#     27: 14,
#     28: 15,
#     30: 16
# }
LIDAR_NAMES = [1, 2, 3, 4, 5]
THING_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
MIN_INST_POINT = 50
INST_DBINFO_PKL = dict()
for c_i in THING_LIST:
    INST_DBINFO_PKL[SemKITTI_label_name_22[c_i]] = []

def prepare_file_list():
    # file_list = os.listdir(os.path.join(DATA_ROOT, 'velodyne'))
    label_files = []
    sensor_files = []

    with open(os.path.join(DATA_ROOT, 'keyframes.txt'), 'r') as f:
        pcd_files = f.read().splitlines()
    for idx, filename in enumerate(pcd_files):
        token_list = filename.split('/')
        pcd_files[idx] = os.path.join(DATA_ROOT, token_list[-3], token_list[-2], token_list[-1])
    for filename in pcd_files:
        label_files.append(filename.replace('lidar', 'label'))
        sensor_files.append(filename.replace('lidar', 'sensor'))
    pcd_info_list = []
    for idx, path in enumerate(tqdm(label_files)):
        info = {}
        info['label_path'] = label_files[idx]
        info['lidar_path'] = pcd_files[idx]
        info['sensor_file'] = sensor_files[idx]
        token_list = str(label_files[idx]).split('/')
        info['lidar_token'] = '%s_%s' % (token_list[-3], token_list[-1][:-4])
        pcd_info_list.append(info)
    return pcd_info_list


def process_one_sequences(info: dict, save_file=True):

    def _load_sensor_mask(sensor_file_path, ri):
        if ri == 1:
            sensor_file_path = str(sensor_file_path).replace('sensor', 'sensor_ri2')
        return np.fromfile(sensor_file_path, dtype=np.uint8) == LIDAR_NAMES[0]

    def _load_pcd(filepath, labelpath, top_m, ri):
        if ri == 1:
            filepath = str(filepath).replace('lidar', 'lidar_ri2')
            labelpath = str(labelpath).replace('label', 'label_ri2')
        pts = np.fromfile(filepath, dtype=np.float32).reshape((-1, 6))
        xyz, i, r, e = pts[:, :3], np.tanh(pts[:, 3]), pts[:, 4], pts[:, 5]
        pts = np.concatenate([xyz, i.reshape([-1, 1]), e.reshape([-1, 1])], axis=-1)
        pts = pts[top_m]
        annot = np.fromfile(labelpath, dtype=np.int32).reshape([-1, 2])
        sem_label = annot[top_m, 1].astype(np.uint32)
        panoptic_label = annot[top_m, 0].astype(np.int32)

        return pts, sem_label, panoptic_label

    lidar_token = info['lidar_token']
    lidar_path = info['lidar_path']
    label_path = info['label_path']
    sensor_file = info['sensor_file']
    ri_list = [0, 1]
    pts_list, sem_label_list,  panoptic_label_list = [], [], []
    for ri in ri_list:
        top_m = _load_sensor_mask(sensor_file, ri)
        pts, sem_label, panoptic_label = _load_pcd(filepath=lidar_path, labelpath=label_path, top_m=top_m, ri=ri)
        pts_list.append(pts)
        sem_label_list.append(sem_label)
        panoptic_label_list.append(panoptic_label)
    point_xyzie = np.concatenate(pts_list, axis=0)
    sem_label =  np.concatenate(sem_label_list, axis=0).astype(np.uint8)
    panoptic_label = np.concatenate(panoptic_label_list, axis=0).astype(np.int32)
    valid_mask = (sem_label != 0)
    point_xyzie = point_xyzie[valid_mask]
    sem_label = sem_label[valid_mask]
    panoptic_label = panoptic_label[valid_mask]
    if not save_file:
        visualize_pcd(point_xyzie[:, :3], predict=np.vectorize(MapWaymo2NUSC.__getitem__)(sem_label), target=panoptic_label)

    for thing_id in THING_LIST:
        thing_mask = np.zeros_like(sem_label, dtype=bool)  # 每个类别一个mask
        thing_mask[sem_label == thing_id] = True  # 所有属于thing_id的类
        panoptic_label_thing = panoptic_label[thing_mask]  # 全景标签存在小于 2^16 的异常点
        unique_inst_label = np.unique(panoptic_label_thing)

        thing_name = SemKITTI_label_name_22[thing_id]
        for uq_inst_label in unique_inst_label:  # 对于每一个实例
            if uq_inst_label == 0 or uq_inst_label == -1:
                continue
            index = np.where(panoptic_label == uq_inst_label)[0]
            if index.shape[0] < MIN_INST_POINT:  # 如果
                continue
            if np.sum(panoptic_label[index]) == 0:
                continue
            if save_file:
                dir_path = os.path.join(DATABASE_SAVE_DIR, SemKITTI_label_name_22[thing_id])
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path)
                    except OSError:
                        print(f'Error occurred when creating ground truth mask dir "{dir_path}".')
                    else:
                        print(f'dir created "{dir_path}.')
                file_path = os.path.join(dir_path, '%s_%s_%s.bin' % (str(lidar_token), str(thing_name), str(uq_inst_label)))
                inst_points = point_xyzie[index, :]
                if not os.path.exists(file_path):
                    inst_points.tofile(file_path)
                    INST_DBINFO_PKL[thing_name].append(file_path)
            else:
                inst_points = point_xyzie[index, :]
                inst_sem_label = sem_label[index]
                inst_pano_label = panoptic_label[index]
                print('num_points:', inst_points.shape[0])
                print('class:', SemKITTI_label_name_22[inst_sem_label[0]])
                if inst_sem_label[0] == 1:
                    continue
                visualize_pcd(inst_points[:, :3], predict=inst_sem_label, target=inst_pano_label)

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

