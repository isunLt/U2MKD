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

        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        num_vox = feed_dict['num_vox']

        keyframe_mask = feed_dict.get('keyframe_mask', None)
        keyframe_mask = keyframe_mask.F.cuda(non_blocking=True) if keyframe_mask is not None else None

        cur = 0
        for n in num_vox:
            pts = feed_dict['lidar'].F[cur:cur+n, :3]
            label = targets[cur:cur+n]
            visualize_pcd(xyz=pts, target=label, select_inds=feed_dict['raw_mask'].F[cur:cur + n])
            # visualize_pcd(xyz=pts, target=label, select_inds=feed_dict['raw_mask'].F[cur:cur+n])
            # visualize_pcd(xyz=pts, target=label, select_inds=feed_dict['mix_mask'].F[cur:cur+n])
            cur += n

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


class NuScenesLCTSDFullTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: dict[str, Callable],
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 seed: int,
                 weight_path: str = None) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = configs.get('workers_per_gpu')
        self.seed = seed
        self.amp_enabled = configs.get('amp_enabled')
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.pretrain_weight = configs['model'].get('pretrain_weight')
        self.teacher_pretrain_weight = configs['model']['teacher_pretrain']
        self.num_classes = configs['data']['num_classes']
        self.ignore_label = configs['data']['ignore_label']
        self.w_kl = configs['criterion']['w_kl']
        self.w_feat = configs['criterion']['w_feat']
        self.non_dist = configs['non_dist']
        print('self.w_kl', self.w_kl)
        self.multisweeps = configs['dataset']['multisweeps']['num_sweeps']
        self.mse_norm_feat = configs['criterion']['mse_norm_feat']

    def _before_train(self) -> None:
        if self.weight_path is not None and os.path.exists(self.weight_path):
            print("load weight from", self.weight_path)
            if self.non_dist:
                self.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))
            else:
                state_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
                self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()})
        elif self.pretrain_weight is not None and os.path.exists(self.pretrain_weight):
            print('load pretrained weight from', self.pretrain_weight)
            state_dict = torch.load(self.pretrain_weight, map_location=torch.device('cpu'))
            new_state_dict = {}
            for k, v in state_dict['model'].items():
                if 'classifier' not in k:
                    # new_state_dict[k.replace('module.', '')] = v
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        elif self.teacher_pretrain_weight is not None and os.path.exists(self.teacher_pretrain_weight):
            print('load teacher weight from', self.teacher_pretrain_weight)
            state_dict = torch.load(self.teacher_pretrain_weight, map_location=torch.device('cpu'))['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            if self.non_dist:
                self.model.module.model_t.load_state_dict(new_state_dict, strict=True)
            else:
                self.model.model_t.load_state_dict(new_state_dict, strict=True)
        else:
            print("train from sketch")

    def _before_epoch(self) -> None:
        self.model.train()
        if self.non_dist:
            self.model.module.model_t.eval()
        else:
            self.model.model_t.eval()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _visualize_sample(self, feed_dict: dict):

        num_vox = feed_dict['num_vox']
        targets = feed_dict['targets'].F.long()

        b, _, h, w, c = feed_dict['images'].size()
        img_feat_tensor = []
        for mask, coord, img in zip(feed_dict['masks'], feed_dict['pixel_coordinates'], feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous()):
            """
                mask <Tensor, [6, N]>
                coord <Tensor, [6, N, 2]>
                img <Tensor, [6, C, H, W]>
            """
            imf = torch.zeros(size=(mask.size(1), img.size(1)))
            imf_list = Feature_Gather(img, coord).permute(0, 2, 1)  # [B, N, C]
            assert mask.size(0) == coord.size(0)
            for idx in range(mask.size(0)):
                imf[mask[idx]] = imf_list[idx, mask[idx], :]
            img_feat_tensor.append(imf)

        # v_co_list, v_t_list = [], []
        # cur = 0
        # for n_p, coord, mask in zip(num_vox, feed_dict['pixel_coordinates'], feed_dict['masks']):
        #     targets_per_pcd = targets[cur:cur+n_p]
        #     for co, ma in zip(coord, mask):
        #         v_co, v_t = co[ma, :], targets_per_pcd[ma]
        #         v_co_list.append(v_co.clone())
        #         v_t_list.append(v_t.clone())
        #     cur += n_p

        # b, _, h, w, c = feed_dict['images'].size()
        # images = feed_dict['images'].view(-1, h, w, c)
        # for image, v_co, v_t in zip(images, v_co_list, v_t_list):
        #     visualize_img(image, point=torch.cat([v_co, v_t.unsqueeze(-1)], dim=-1))

        cur = 0
        for n, color in zip(num_vox, img_feat_tensor):
            pts = feed_dict['lidar'].F[cur:cur+n, :3]
            label = targets[cur:cur+n]
            visualize_pcd(xyz=pts, rgb=color, target=label, select_inds=~feed_dict['inst_aug_mask'].F[cur:cur+n])
            cur += n

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        def _prepare_input(feed_dict: Dict[str, Any]) -> Dict[str, Any]:
            in_mod = {}
            for k, v in feed_dict.items():
                if k == 'lidar':
                    in_mod[k] = v.cuda()
                elif k == 'images':
                    in_mod[k] = v.permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
                elif k == 'pixel_coordinates':
                    in_mod[k] = [coord.cuda(non_blocking=True) for coord in v]
                elif k == 'masks':
                    in_mod[k] = [mask.cuda(non_blocking=True) for mask in v]
                elif k == 'fov_mask':
                    in_mod[k] = v.F.cuda(non_blocking=True)
            return in_mod

        # torch.cuda.empty_cache()

        in_mod = {}
        in_mod['student'] = _prepare_input(feed_dict['feed_dict_s'])
        in_mod['teacher'] = _prepare_input(feed_dict['feed_dict_t'])
        targets = feed_dict['feed_dict_s']['targets'].F.long().cuda(non_blocking=True)
        inv_map = feed_dict['feed_dict_t']['inverse_map'].F.cuda(non_blocking=True)
        fov_mask = feed_dict['feed_dict_s']['fov_mask'].F.cuda(non_blocking=True)
        # targets_t = feed_dict['feed_dict_t']['targets'].F.long().cuda(non_blocking=True)
        # self._visualize_sample(feed_dict['feed_dict_s'])
        keyframe_mask_full = feed_dict['feed_dict_t'].get('keyframe_mask_full', None)
        keyframe_mask_full = keyframe_mask_full.F.cuda(non_blocking=True) if keyframe_mask_full is not None else None

        with amp.autocast(enabled=self.amp_enabled):
            outputs = self.model(in_mod)
            if self.model.training:
                num_pts = feed_dict['feed_dict_t']['num_pts']
                num_vox = feed_dict['feed_dict_t']['num_vox']
                inds_s = feed_dict['feed_dict_s']['inds']
                x_vox_t = outputs['t']['x_vox']
                # x_vox_targets = []
                x_vox_t2s = []
                cur_v, cur_p = 0, 0
                for b_i, (np, nv, inds) in enumerate(zip(num_pts, num_vox, inds_s)):
                    tmp = x_vox_t[cur_v:cur_v + nv]
                    inv = inv_map[cur_p:cur_p + np]
                    if keyframe_mask_full is not None:
                        kfm = keyframe_mask_full[cur_p: cur_p + np]
                        x_vox_t2s.append(tmp[inv, :][kfm, :][inds[0], :])
                    else:
                        x_vox_t2s.append(tmp[inv, :][inds[0], :])
                    # t = targets_t[cur_v:cur_v + nv]
                    # x_vox_targets.append(t[inv][kfm][inds[0]])

                    cur_v += nv
                    cur_p += np
                x_vox_t2s = torch.cat(x_vox_t2s, dim=0)
                # x_vox_targets = torch.cat(x_vox_targets, dim=0)

                feat_t = outputs['t']['pts_feats'][0]
                feat_t2s = []
                cur_v, cur_p = 0, 0
                for b_i, (np, nv, inds) in enumerate(zip(num_pts, num_vox, inds_s)):
                    tmp = feat_t[cur_v:cur_v + nv, :]
                    inv = inv_map[cur_p:cur_p + np]
                    if keyframe_mask_full is not None:
                        kfm = keyframe_mask_full[cur_p: cur_p + np]
                        feat_t2s.append(tmp[inv, :][kfm, :][inds[0], :])
                    else:
                        feat_t2s.append(tmp[inv, :][inds[0], :])
                    cur_v += nv
                    cur_p += np
                feat_t2s = torch.cat(feat_t2s, dim=0)

                x_vox = outputs['stu']['x_vox']
                x_pix = outputs['stu']['x_pix']
                # pred_t = torch.argmax(F.softmax(x_vox_t, dim=1), dim=1)
                # acc = torch.sum(x_vox_targets == targets) / targets.size(0)
                # acc_t = torch.sum(pred_t == targets_t) / targets_t.size(0)
                loss_dict = {}
                loss_dict['ce_vox'] = self.criterion['lovasz'](x_vox, targets)
                loss_dict['ce_pix'] = self.criterion['lovasz'](x_pix[fov_mask], targets[fov_mask])
                loss_dict['kl'] = self.criterion['kl'](F.log_softmax(x_vox, dim=1), F.softmax(x_vox_t2s.detach(), dim=1))
                loss_dict['mse'] = outputs['stu']['mse_loss']

                pts_feat_s = outputs['stu']['pts_feats'][0]
                if self.mse_norm_feat:
                    f_max = torch.max(pts_feat_s, dim=-1, keepdim=True).values
                    f_min = torch.min(pts_feat_s, dim=-1, keepdim=True).values
                    pts_feat_s = (pts_feat_s - f_min) / (f_max - f_min)
                    f_max = torch.max(feat_t2s, dim=-1, keepdim=True).values
                    f_min = torch.min(feat_t2s, dim=-1, keepdim=True).values
                    feat_t2s = (feat_t2s - f_min) / (f_max - f_min)
                loss_dict['feat'] = self.criterion['mse'](pts_feat_s, feat_t2s.detach())
        if self.model.training:
            ce_vox = loss_dict.get('ce_vox')
            ce_pix = loss_dict.get('ce_pix')
            kl_vox = loss_dict.get('kl')
            self.summary.add_scalar('ce/vox', ce_vox.item())
            self.summary.add_scalar('ce/pix', ce_pix.item())
            self.summary.add_scalar('ce/kl', kl_vox.item())
            loss = ce_vox + ce_pix + self.w_kl * kl_vox
            for idx, mse in enumerate(loss_dict['mse']):
                self.summary.add_scalar('mse/layer%s' % str(idx), mse.item())
                loss += mse
            self.summary.add_scalar('mse/feat', loss_dict['feat'].item())
            loss += self.w_feat * loss_dict['feat']

            self.summary.add_scalar('total_loss', loss.item())
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            return outputs
        else:
            invs = feed_dict['feed_dict_s']['inverse_map']
            all_labels = feed_dict['feed_dict_s']['targets_mapped']
            fov_labels = feed_dict['feed_dict_s']['label_fov']
            _outputs_vox, _outputs_pix = [], []
            _targets, _targets_fov = [], []
            outputs_vox = outputs['stu']['x_vox']
            outputs_pix = outputs['stu']['x_pix']
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (in_mod['student']['lidar'].C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv].argmax(1)
                outputs_mapped_pix = outputs_pix[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                targets_mapped_fov = fov_labels.F[cur_label]
                _outputs_vox.append(outputs_mapped_vox)
                _outputs_pix.append(outputs_mapped_pix)
                _targets.append(targets_mapped)
                _targets_fov.append(targets_mapped_fov)
            outputs_vox = torch.cat(_outputs_vox, 0).cpu()
            outputs_pix = torch.cat(_outputs_pix, 0).cpu()
            targets = torch.cat(_targets, 0).cpu()
            targets_fov = torch.cat(_targets_fov, dim=0).cpu()
            ret_dict = {}
            ret_dict['outputs_vox'] = outputs_vox
            ret_dict['outputs_pix'] = outputs_pix
            ret_dict['targets'] = targets
            ret_dict['targets_fov'] = targets_fov
            if configs['debug']['debug_val']:
                invs = feed_dict['feed_dict_t']['inverse_map']
                all_labels = feed_dict['feed_dict_t']['targets_mapped']
                _outputs_vox = []
                _targets = []
                outputs_vox_t = outputs['t']['x_vox']
                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (in_mod['teacher']['lidar'].C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped_vox = outputs_vox_t[cur_scene_pts][cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs_vox.append(outputs_mapped_vox)
                    _targets.append(targets_mapped)
                outputs_vox_t = torch.cat(_outputs_vox, 0).cpu()
                targets_t = torch.cat(_targets, 0).cpu()
                if self.multisweeps != 0:
                    keyframe_mask_full = feed_dict['feed_dict_t']['keyframe_mask_full'].F
                    outputs_vox_t = outputs_vox_t[keyframe_mask_full]
                    targets_t = targets_t[keyframe_mask_full]
                ret_dict['outputs_vox_t'] = outputs_vox_t
                ret_dict['targets_t'] = targets_t
            return ret_dict

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


class NuScenes_Evaluator(Trainer):
    def __init__(self,
                 model: nn.Module,
                 num_workers: int,
                 seed: int,
                 weight_path: str = None,
                 amp_enabled: bool = False,
                 ignore_label: int = 0) -> None:

        self.model = model
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)

        self.epoch_num = 1
        self.weight_path = weight_path
        self.ignore_label = ignore_label
        self.num_vote = configs['dataset']['num_vote']
        # self.num_total = 0
        # self.num_sub = 0

    def run_miou(self,
                 dataflow: torch.utils.data.DataLoader,
                 *,
                 num_epochs: int = 1,
                 callbacks: Optional[List[Callback]] = None
                 ) -> None:
        if callbacks is None:
            callbacks = []
        callbacks += [
            ProgressBar(),
        ]
        self.train(dataflow=dataflow,
                   num_epochs=num_epochs,
                   callbacks=callbacks)

    def _before_train(self) -> None:

        assert self.weight_path is not None and os.path.exists(self.weight_path)
        print("load weight from", self.weight_path)
        state_dict = torch.load(self.weight_path, map_location=torch.device('cpu'))
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()})
        # self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu'))['model'])

    def _before_epoch(self) -> None:
        self.model.eval()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:

        # feed_dict = feed_dict['feed_dict_t']
        in_mod = {}
        in_mod['lidar'] = feed_dict['lidar'].cuda()  # [x, y, z, batch_idx] batch_idx表示这个voxel属于哪个batch
        in_mod['images'] = feed_dict['images'].permute(0, 1, 4, 2, 3).contiguous().cuda(non_blocking=True)
        in_mod['pixel_coordinates'] = [coord.cuda() for coord in feed_dict['pixel_coordinates']]
        in_mod['masks'] = [mask.cuda() for mask in feed_dict['masks']]
        # targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        lidar_token = feed_dict['lidar_token']
        if configs['dataset']['num_vote'] > 1:
            lidar_token = lidar_token[0]
        # view_mask = [m.cuda() for m in feed_dict['pt_with_img_idx']]
        # num_pts = [coord.size(1) for coord in feed_dict['pixel_coordinates']]

        # self.num_total += np.sum(np.array(feed_dict['num_total']))
        # self.num_sub += np.sum(np.array(feed_dict['num_sub']))
        #
        # print('sub rate:', float(self.num_sub) / self.num_total)

        with amp.autocast(enabled=self.amp_enabled):
            with torch.no_grad():
                outputs = self.model(in_mod)
                outputs_vox = outputs.get('x_vox')
                # invs = feed_dict['inverse_map'].to(outputs_vox.device, non_blocking=True)
                # all_labels = feed_dict['targets_mapped'].to(outputs_vox.device, non_blocking=True)
                invs = feed_dict['inverse_map']
                all_labels = feed_dict['targets_mapped']
                _outputs_vox, _targets = [], []
                # outputs_vox = outputs.get('x_pix')
                num_pts = []
                for idx in range(invs.C[:, -1].max() + 1):
                    cur_scene_pts = (feed_dict['lidar'].C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    # cur_scene_pts = (feed_dict['lidar'].C[:, -1] == idx)
                    # cur_inv = invs.F[invs.C[:, -1] == idx]
                    # cur_label = (all_labels.C[:, -1] == idx)
                    outputs_mapped_vox = outputs_vox[cur_scene_pts][cur_inv]
                    num_pts.append(outputs_mapped_vox.size(0))
                    targets_mapped = all_labels.F[cur_label]
                    _outputs_vox.append(outputs_mapped_vox)
                    _targets.append(targets_mapped)
                if self.num_vote > 1:
                    outputs_vox = torch.stack(_outputs_vox, 0)
                    outputs_vox = torch.sum(outputs_vox, dim=0)
                    outputs_vox = torch.argmax(outputs_vox, dim=-1).cpu()
                    targets = _targets[0]
                    num_pts = num_pts[0]
                else:
                    outputs_vox = torch.cat(_outputs_vox, 0).cpu()
                    outputs_vox = torch.argmax(outputs_vox, dim=-1)
                    targets = torch.cat(_targets, 0).cpu()
                return {
                    'outputs_vox': outputs_vox,
                    'targets': targets,
                    'lidar_token_list': lidar_token,
                    'num_pts': num_pts
                }


