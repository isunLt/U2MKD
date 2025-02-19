import os
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
import torch.nn.functional as F
import pickle

from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from torchpack.callbacks import Callback, ProgressBar
from torchpack.utils.config import configs
from torchpack.environ import get_run_dir

from core.models.fusion_blocks import Feature_Gather
from visualize_utils import visualize_img, visualize_pcd


class NuScenesTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.ignore_label = configs['criterion']['ignore_index']
        self.multisweeps = configs['dataset']['multisweeps']['num_sweeps']

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        torch.cuda.empty_cache()
        #
        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        num_vox = feed_dict['num_vox']

        keyframe_mask = feed_dict.get('keyframe_mask', None)
        keyframe_mask = keyframe_mask.F.cuda(non_blocking=True) if keyframe_mask is not None else None

        # cur = 0
        # for n in num_vox:
        #     pts = feed_dict['lidar'].F[cur:cur+n, :3]
        #     label = targets[cur:cur+n]
        #     # visualize_pcd(xyz=pts, target=label, select_inds=feed_dict['raw_mask'].F[cur:cur+n])
        #     visualize_pcd(xyz=pts, target=label)
        #     cur += n

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if self.model.training:
                if keyframe_mask is not None:
                    outputs['x_vox'] = outputs['x_vox'][keyframe_mask]
                    targets = targets[keyframe_mask]
                loss_dict = self.criterion(outputs['x_vox'], targets)
        if self.model.training:
            predict_vox = loss_dict
            self.summary.add_scalar('ce/vox', predict_vox.item())
            loss = predict_vox
            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
            _outputs_vox = []
            _targets = []
            outputs_vox = outputs.get('x_vox')
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _targets.append(targets_mapped)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            if self.multisweeps != 0:
                keyframe_mask_full = feed_dict['keyframe_mask_full'].F
                outputs_vox = outputs_vox[keyframe_mask_full]
                targets = targets[keyframe_mask_full]
            return {
                'outputs_vox': outputs_vox,
                'targets': targets,
            }

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass
