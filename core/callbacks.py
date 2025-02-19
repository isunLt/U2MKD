import json
import os
import numpy as np
from typing import Any, Dict, Optional

import torch

from torchpack.environ import get_run_dir
from torchpack import distributed as dist
from torchpack.callbacks import TFEventWriter
from torchpack.callbacks.callback import Callback
from torchpack.utils import fs, io
from torchpack.utils.logging import logger
from nuscenes.eval.lidarseg.utils import ConfusionMatrix

from prettytable import PrettyTable

__all__ = ['MeanIoU', 'EpochSaver']


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


class MeanIoU(Callback):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.mapclass2name = None
        if self.num_classes == 17:
            self.mapclass2name = SemKITTI_label_name_16
        elif self.num_classes == 20:
            self.mapclass2name = SemKITTI_label_name_19
        elif self.num_classes == 23:
            self.mapclass2name = SemKITTI_label_name_22

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        if type(outputs) != np.ndarray:
            for i in range(self.num_classes):
                self.total_seen[i] += torch.sum(targets == i).item()
                self.total_correct[i] += torch.sum(
                    (targets == i) & (outputs == targets)).item()
                self.total_positive[i] += torch.sum(
                    outputs == i).item()
        else:
            for i in range(self.num_classes):
                self.total_seen[i] += np.sum(targets == i)
                self.total_correct[i] += np.sum((targets == i)
                                                & (outputs == targets))
                self.total_positive[i] += np.sum(outputs == i)

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i], reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i], reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i], reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                if i == self.ignore_label:
                    continue
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i] +
                                                   self.total_positive[i] -
                                                   self.total_correct[i])
                ious.append(cur_iou)

        miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
            for writer in self.trainer.summary.writers:
                if isinstance(writer, TFEventWriter):
                    for idx in range(1, self.num_classes):
                        writer.add_scalar(self.name + '/' + self.mapclass2name[idx], ious[idx-1] * 100)
            pt = PrettyTable()
            pt.field_names = ['Item'] + list(self.mapclass2name.values())[1:] + ['Mean']
            pt.add_row(['IoU'] + [round(i * 100, 2) for i in ious] + [round(miou * 100, 2)])
            print(pt)
        else:
            pt = PrettyTable()
            pt.field_names = ['Item'] + list(self.mapclass2name.values())[1:] + ['Mean']
            pt.add_row(['IoU'] + [round(i * 100, 2) for i in ious] + [round(miou * 100, 2)])
            print(pt)


class EpochSaver(Callback):
    """
    Save the checkpoint once triggered.
    """
    master_only: bool = True

    def __init__(self, *, epoch_to_save: int = 5,
                 save_dir: Optional[str] = None) -> None:
        self.epoch_to_save = epoch_to_save
        if save_dir is None:
            save_dir = os.path.join(get_run_dir(), 'checkpoints')
        self.save_dir = fs.normpath(save_dir)

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        if self.trainer.epoch_num and not (self.trainer.epoch_num % self.epoch_to_save):
            save_path = os.path.join(self.save_dir,
                                     f'epoch-{self.trainer.epoch_num}.pt')
            try:
                io.save(save_path, self.trainer.state_dict())
            except OSError:
                logger.exception(
                    f'Error occurred when saving checkpoint "{save_path}".')
            else:
                logger.info(f'Checkpoint saved: "{save_path}".')


class InferTime(Callback):
    def __init__(self,
                 batchsize: int) -> None:
        self.batchsize = batchsize

    def _before_epoch(self) -> None:
        self.infer_time_list = []

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        infer_t = output_dict['infer_time']
        self.infer_time_list.append(infer_t)

    def _after_epoch(self) -> None:
        self.infer_time_list = dist.allgather(self.infer_time_list)
        self.infer_time_list = [a[10:-3] for a in self.infer_time_list]
        # print('infer_time_list:', self.infer_time_list)
        m_infer_time = np.mean(self.infer_time_list) / self.batchsize
        print('infer time:', m_infer_time * 1000)