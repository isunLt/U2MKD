from typing import Callable

import torch
import torch.optim
from torch import nn
import torchpack.distributed as dist
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

import numpy as np

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset(dataset_name: str = None, **kwargs) -> dict:
    if dataset_name is None:
        dataset_name = configs.dataset.name
    if dataset_name == 'semantic_kitti':
        from core.datasets.semantic_kitti import SemanticKITTI
        dataset = SemanticKITTI(root=configs.dataset.root,
                                voxel_size=configs.dataset.voxel_size)
    elif dataset_name == 'semantic_nusc':
        from core.datasets.semantic_nusc import NuScenes
        dataset = NuScenes(root=configs.dataset.root,
                           voxel_size=configs.dataset.voxel_size,
                           version="v1.0-trainval",
                           verbose=True)
    # elif dataset_name == 'semantic_nusc_2d':
    #     from core.datasets.semantic_nusc_2d import NuScenes2D
    #     dataset = NuScenes2D(root=configs.dataset.root,
    #                          voxel_size=configs.dataset.voxel_size,
    #                          version="v1.0-trainval",
    #                          verbose=True,
    #                          image_crop_rate=configs.dataset.im_cr)
    # elif dataset_name == 'lc_semantic_kitti':
    #     from core.datasets import LCSemanticKITTI
    #     dataset = LCSemanticKITTI(root=configs.dataset.root,
    #                               voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_kitti_distill':
    #     from core.datasets.lc_semantic_kitti_distill import LCSemanticKITTIDistill
    #     dataset = LCSemanticKITTIDistill(root=configs.dataset.root,
    #                                      voxel_size=configs.dataset.voxel_size,
    #                                      config=configs)
    # elif dataset_name == 'lc_semantic_kitti_full':
    #     from core.datasets.lc_semantic_kitti_full import LCSemanticKITTIFull
    #     dataset = LCSemanticKITTIFull(root=configs.dataset.root,
    #                                   voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_kitti_full_tta':
    #     from core.datasets.lc_semantic_kitti_full_tta import LCSemanticKITTIFullTTA
    #     dataset = LCSemanticKITTIFullTTA(root=configs.dataset.root,
    #                                      voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_kitti_tsd_full':
    #     from core.datasets.lc_semantic_kitti_tsd_full import LCSemanticKITTITSDFull
    #     dataset = LCSemanticKITTITSDFull(root=configs.dataset.root,
    #                                   voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_nusc':
    #     from core.datasets.lc_semantic_nusc import LCNuScenes
    #     dataset = LCNuScenes(root=configs.dataset.root,
    #                          voxel_size=configs.dataset.voxel_size,
    #                          version="v1.0-trainval",
    #                          verbose=True,
    #                          im_cr=configs.dataset.im_cr)
    # elif dataset_name == 'lc_semantic_nusc_full':
    #     from core.datasets.lc_semantic_nusc_full import LCNuScenesFull
    #     dataset = LCNuScenesFull(root=configs.dataset.root,
    #                              voxel_size=configs.dataset.voxel_size,
    #                              version="v1.0-trainval",
    #                              verbose=True,
    #                              im_cr=configs.dataset.im_cr)
    # elif dataset_name == 'lc_semantic_nusc_tta':
    #     from core.datasets.lc_semantic_nusc_tta import LCNuScenesTTA
    #     dataset = LCNuScenesTTA(root=configs.dataset.root,
    #                             voxel_size=configs.dataset.voxel_size,
    #                             version="v1.0-trainval",
    #                             verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_full_tta':
    #     from core.datasets.lc_semantic_nusc_full_tta import LCNuScenesFullTTA
    #     dataset = LCNuScenesFullTTA(root=configs.dataset.root,
    #                                 voxel_size=configs.dataset.voxel_size,
    #                                 version=configs.dataset.version,
    #                                 verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_distill':
    #     from core.datasets.lc_semantic_nusc_distill import LCNuScenesDistill
    #     dataset = LCNuScenesDistill(root=configs.dataset.root,
    #                                 voxel_size=configs.dataset.voxel_size,
    #                                 version="v1.0-trainval",
    #                                 verbose=True,
    #                                 config=configs)
    # elif dataset_name == 'semantic_nusc_tsd':
    #     from core.datasets.semantic_nusc_tsd import NuScenesTSDistill
    #     dataset = NuScenesTSDistill(root=configs.dataset.root,
    #                                 voxel_size=configs.dataset.voxel_size,
    #                                 version="v1.0-trainval",
    #                                 verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_tsd':
    #     from core.datasets.lc_semantic_nusc_tsd import LCNuScenesTSDistill
    #     dataset = LCNuScenesTSDistill(root=configs.dataset.root,
    #                                   voxel_size=configs.dataset.voxel_size,
    #                                   version="v1.0-trainval",
    #                                   verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_tsd_l2lc':
    #     from core.datasets.lc_semantic_nusc_tsd_l2lc import LCNuScenesL2LCTSDistill
    #     dataset = LCNuScenesL2LCTSDistill(root=configs.dataset.root,
    #                                       voxel_size=configs.dataset.voxel_size,
    #                                       version="v1.0-trainval",
    #                                       verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_tsd_lc2c':
    #     from core.datasets.lc_semantic_nusc_tsd_lc2c import LCNuScenesTSDistillLC2C
    #     dataset = LCNuScenesTSDistillLC2C(root=configs.dataset.root,
    #                                       voxel_size=configs.dataset.voxel_size,
    #                                       version="v1.0-trainval",
    #                                       verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_tsd_c2lc':
    #     from core.datasets.lc_semantic_nusc_tsd_c2lc import LCNuScenesTSDistillC2LC
    #     dataset = LCNuScenesTSDistillC2LC(root=configs.dataset.root,
    #                                       voxel_size=configs.dataset.voxel_size,
    #                                       version="v1.0-trainval",
    #                                       verbose=True)
    elif dataset_name == 'lc_semantic_nusc_tsd_full':
        from core.datasets.lc_semantic_nusc_tsd_full import LCNuScenesTSDistillFull
        dataset = LCNuScenesTSDistillFull(root=configs.dataset.root,
                                          voxel_size=configs.dataset.voxel_size,
                                          version="v1.0-trainval",
                                          verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_tsd_full_umt':
    #     from core.datasets.lc_semantic_nusc_tsd_full_umt import LCNuScenesTSDistillFull
    #     dataset = LCNuScenesTSDistillFull(root=configs.dataset.root,
    #                                       voxel_size=configs.dataset.voxel_size,
    #                                       version="v1.0-trainval",
    #                                       verbose=True)
    # elif dataset_name == 'lc_semantic_nusc_tsd_full_lc2l':
    #     from core.datasets.lc_semantic_nusc_tsd_full_lc2l import LCNuScenesTSDistillLC2LFull
    #     dataset = LCNuScenesTSDistillLC2LFull(root=configs.dataset.root,
    #                                           voxel_size=configs.dataset.voxel_size,
    #                                           version="v1.0-trainval",
    #                                           verbose=True)
    # elif dataset_name == 'semantic_waymo':
    #     from core.datasets.semantic_waymo import SemanticWaymo
    #     dataset = SemanticWaymo(root=configs.dataset.root,
    #                             voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'semantic_waymo_tta':
    #     from core.datasets.semantic_waymo_tta import SemanticWaymoTTA
    #     dataset = SemanticWaymoTTA(root=configs.dataset.root,
    #                                voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_waymo':
    #     from core.datasets.lc_semantic_waymo import LCSemanticWaymo
    #     dataset = LCSemanticWaymo(root=configs.dataset.root,
    #                               voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_waymo_full':
    #     from core.datasets.lc_semantic_waymo_full import LCSemanticWaymoFull
    #     dataset = LCSemanticWaymoFull(root=configs.dataset.root,
    #                                   voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_waymo_tsd_full':
    #     from core.datasets.lc_semantic_waymo_tsd_full import LCSemanticWaymoFullTSD
    #     dataset = LCSemanticWaymoFullTSD(root=configs.dataset.root,
    #                                      voxel_size=configs.dataset.voxel_size)
    # elif dataset_name == 'lc_semantic_waymo_full_tta':
    #     from core.datasets.lc_semantic_waymo_full_tta import LCSemanticWaymoFullTTA
    #     dataset = LCSemanticWaymoFullTTA(root=configs.dataset.root,
    #                                      voxel_size=configs.dataset.voxel_size)
    else:
        raise NotImplementedError(dataset_name)
    return dataset


def make_model(model_name=None) -> nn.Module:
    if model_name is None:
        model_name = configs.model.name
    if "cr" in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    if model_name == 'spvcnn':
        from core.models.semantickitti.spvcnn import SPVCNN
        model = SPVCNN(
            in_channel=configs.model.in_channel,
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size
        )
    # elif model_name == 'swiftnet_only':
    #     from core.models.nuscenes.swiftnet_only import SWIFTNET_ONLY
    #     model = SWIFTNET_ONLY(
    #         in_channel=configs.model.in_channel,
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain
    #     )
    # elif model_name == 'spvcnn_swiftnet34':
    #     from core.models.semantickitti.spvcnn_swiftnet34 import SPVCNN_SWIFTNET34
    #     model = SPVCNN_SWIFTNET34(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18':
    #     from core.models.nuscenes.spvcnn_swiftnet18 import SPVCNN_SWIFTNET18
    #     model = SPVCNN_SWIFTNET18(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_2dlearner':
    #     from core.models.nuscenes.spvcnn_swiftnet18_2dlearner import SPVCNN_SWIFTNET18_2DLEARNER
    #     model = SPVCNN_SWIFTNET18_2DLEARNER(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet34_distill':
    #     from core.models.semantickitti.spvcnn_swiftnet34_distill import SPVCNN_SWIFTNET34_DISTILL
    #     model = SPVCNN_SWIFTNET34_DISTILL(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_distill':
    #     from core.models.nuscenes.spvcnn_swiftnet18_distill import SPVCNN_SWIFTNET18_DISTILL
    #     model = SPVCNN_SWIFTNET18_DISTILL(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd import SPVCNN_SWIFTNET18_TSD
    #     model = SPVCNN_SWIFTNET18_TSD(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_c2l':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_c2l import SPVCNN_SWIFTNET18_TSD_C2L
    #     model = SPVCNN_SWIFTNET18_TSD_C2L(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_l2lc':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_l2lc import SPVCNN_SWIFTNET18_TSD_L2LC
    #     model = SPVCNN_SWIFTNET18_TSD_L2LC(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_c2lc':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_c2lc import SPVCNN_SWIFTNET18_TSD_C2LC
    #     model = SPVCNN_SWIFTNET18_TSD_C2LC(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_full':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_full import SPVCNN_SWIFTNET18_TSD_FULL
    #     model = SPVCNN_SWIFTNET18_TSD_FULL(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_full_feat':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_full_feat import SPVCNN_SWIFTNET18_TSD_FULL_FEAT
    #     model = SPVCNN_SWIFTNET18_TSD_FULL_FEAT(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_full_feat_v2':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_full_feat_v2 import SPVCNN_SWIFTNET18_TSD_FULL_FEAT_V2
    #     model = SPVCNN_SWIFTNET18_TSD_FULL_FEAT_V2(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_full_feat_v3':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_full_feat_v3 import SPVCNN_SWIFTNET18_TSD_FULL_FEAT_V3
    #     # model = SPVCNN_SWIFTNET18_TSD_FULL_FEAT_V3(
    #     #     num_classes=configs.data.num_classes,
    #     #     cr=cr,
    #     #     pres=configs.dataset.voxel_size,
    #     #     vres=configs.dataset.voxel_size,
    #     #     imagenet_pretrain=configs.model.imagenet_pretrain,
    #     #     debug_val=configs.debug.debug_val
    #     # )
    #     voxel_size = configs.dataset.voxel_size
    #     if not isinstance(voxel_size, list):
    #         voxel_size_list = [voxel_size] * 3
    #     else:
    #         voxel_size_list = voxel_size
    #     patch_size = np.array([voxel_size_list[i] * configs.model.patch_size for i in range(3)]).astype(
    #         np.float32)  # [0.1, 0.1, 0.1]
    #     window_size = patch_size * configs.model.window_size
    #     window_size_sphere = np.array(configs.model.window_size_sphere)
    #     model = SPVCNN_SWIFTNET18_TSD_FULL_FEAT_V3(
    #         window_size=window_size,
    #         window_size_sphere=configs.model.window_size_sphere,
    #         quant_size=window_size / configs.model.quant_size_scale,
    #         quant_size_sphere=window_size_sphere / configs.model.quant_size_scale,
    #         drop_path_rate=configs.model.drop_path_rate,
    #         window_size_scale=configs.model.window_size_scale,
    #         a=configs.model.a,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size
    #     )
    # elif model_name == 'spvcnn_swiftnet34_tsd_full_feat_v2':
    #     from core.models.nuscenes.spvcnn_swiftnet34_tsd_full_feat_v2 import SPVCNN_SWIFTNET34_TSD_FULL_FEAT_V2
    #     model = SPVCNN_SWIFTNET34_TSD_FULL_FEAT_V2(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_full_lc2l':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_full_lc2l import SPVCNN_SWIFTNET18_TSD_FULL_LC2L
    #     model = SPVCNN_SWIFTNET18_TSD_FULL_LC2L(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_full_lc2l_wlearner':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_full_lc2l_wlearner import SPVCNN_SWIFTNET18_TSD_FULL_LC2L_wLEARNER
    #     model = SPVCNN_SWIFTNET18_TSD_FULL_LC2L_wLEARNER(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_lc2c':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_lc2c import SPVCNN_SWIFTNET18_TSD_LC2L
    #     model = SPVCNN_SWIFTNET18_TSD_LC2L(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_feat':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_feat import SPVCNN_SWIFTNET18_TSD_FEAT
    #     model = SPVCNN_SWIFTNET18_TSD_FEAT(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_tsd_feat':
    #     from core.models.nuscenes.spvcnn_tsd_feat import SPVCNN_TSD_FEAT
    #     model = SPVCNN_TSD_FEAT(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_tsd_train':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_train import SPVCNN_SWIFTNET18_TSD_TRAIN
    #     model = SPVCNN_SWIFTNET18_TSD_TRAIN(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #         debug_val=configs.debug.debug_val
    #     )
    # elif model_name == 'spvcnn_swiftnet18_bifusion':
    #     from core.models.nuscenes.spvcnn_swiftnet18_bifusion import SPVCNN_SWIFTNET18_BIFUSION
    #     model = SPVCNN_SWIFTNET18_BIFUSION(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_bifusion_2dlearner':
    #     from core.models.nuscenes.spvcnn_swiftnet18_bifusion_2dlearner import SPVCNN_SWIFTNET18_BIFUSION_2DLEARNER
    #     model = SPVCNN_SWIFTNET18_BIFUSION_2DLEARNER(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_bifusion_sparse':
    #     from core.models.nuscenes.spvcnn_swiftnet18_bifusion_sparse import SPVCNN_SWIFTNET18_BIFUSION_SPARSE
    #     model = SPVCNN_SWIFTNET18_BIFUSION_SPARSE(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_bifusion_cascade':
    #     from core.models.nuscenes.spvcnn_swiftnet18_bifusion_cascade import SPVCNN_SWIFTNET18_BIFUSION_CASCADE
    #     model = SPVCNN_SWIFTNET18_BIFUSION_CASCADE(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_bifusion_cascade_2dlearner':
    #     from core.models.nuscenes.spvcnn_swiftnet18_bifusion_cascade_2dlearner import SPVCNN_SWIFTNET18_BIFUSION_CASCADE_2DLEARNER
    #     model = SPVCNN_SWIFTNET18_BIFUSION_CASCADE_2DLEARNER(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_bifusion_cascade_2dlearner_fai':
    #     from core.models.nuscenes.spvcnn_swiftnet18_bifusion_cascade_2dlearner_fai import SPVCNN_SWIFTNET18_BIFUSION_CASCADE_2DLEARNER_FAI
    #     model = SPVCNN_SWIFTNET18_BIFUSION_CASCADE_2DLEARNER_FAI(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_cross_learner':
    #     from core.models.nuscenes.spvcnn_swiftnet18_cross_learner import SPVCNN_SWIFTNET18_CROSS_LEARNER
    #     model = SPVCNN_SWIFTNET18_CROSS_LEARNER(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_bfr':
    #     from core.models.nuscenes.spvcnn_swiftnet18_bfr import SPVCNN_SWIFTNET18_BFR
    #     model = SPVCNN_SWIFTNET18_BFR(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_umt_full':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_umt import SPVCNN_SWIFTNET18_UMT_FULL
    #     model = SPVCNN_SWIFTNET18_UMT_FULL(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spvcnn_swiftnet18_fa_full':
    #     from core.models.nuscenes.spvcnn_swiftnet18_tsd_fa import SPVCNN_SWIFTNET18_FA_FULL
    #     model = SPVCNN_SWIFTNET18_FA_FULL(
    #         num_classes=configs.data.num_classes,
    #         cr=cr,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size,
    #         imagenet_pretrain=configs.model.imagenet_pretrain,
    #     )
    # elif model_name == 'spformer':
    #     from core.models.sphereformer.unet_spherical_transformer import Semantic
    #     voxel_size = configs.dataset.voxel_size
    #     if not isinstance(voxel_size, list):
    #         voxel_size_list = [voxel_size] * 3
    #     else:
    #         voxel_size_list = voxel_size
    #     patch_size = np.array([voxel_size_list[i] * configs.model.patch_size for i in range(3)]).astype(
    #         np.float32)  # [0.1, 0.1, 0.1]
    #     window_size = patch_size * configs.model.window_size
    #     window_size_sphere = np.array(configs.model.window_size_sphere)
    #     model = Semantic(
    #         input_c=configs.model.in_channel,  # 4
    #         m=configs.model.m,  # 32
    #         classes=configs.data.num_classes,
    #         block_reps=configs.model.block_reps,
    #         block_residual=configs.model.block_residual,
    #         layers=configs.model.layers,
    #         window_size=window_size,
    #         window_size_sphere=configs.model.window_size_sphere,
    #         quant_size=window_size / configs.model.quant_size_scale,
    #         quant_size_sphere=window_size_sphere / configs.model.quant_size_scale,
    #         rel_query=configs.model.rel_query,
    #         rel_key=configs.model.rel_key,
    #         rel_value=configs.model.rel_value,
    #         drop_path_rate=configs.model.drop_path_rate,
    #         window_size_scale=configs.model.window_size_scale,
    #         grad_checkpoint_layers=configs.model.grad_checkpoint_layers,
    #         sphere_layers=configs.model.sphere_layers,
    #         a=configs.model.a,
    #     )
    elif model_name == 'spvcnn_spformer':
        from core.models.nuscenes.spvcnn_spformer import SPVCNN_SPFORMER
        voxel_size = configs.dataset.voxel_size
        if not isinstance(voxel_size, list):
            voxel_size_list = [voxel_size] * 3
        else:
            voxel_size_list = voxel_size
        patch_size = np.array([voxel_size_list[i] * configs.model.patch_size for i in range(3)]).astype(
            np.float32)  # [0.1, 0.1, 0.1]
        window_size = patch_size * configs.model.window_size
        window_size_sphere = np.array(configs.model.window_size_sphere)
        model = SPVCNN_SPFORMER(
            window_size=window_size,
            window_size_sphere=configs.model.window_size_sphere,
            quant_size=window_size / configs.model.quant_size_scale,
            quant_size_sphere=window_size_sphere / configs.model.quant_size_scale,
            drop_path_rate=configs.model.drop_path_rate,
            window_size_scale=configs.model.window_size_scale,
            a=configs.model.a,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size
        )
    # elif model_name == 'spvcnn_swiftnet18_spformer':
    #     from core.models.nuscenes.spvcnn_swiftnet18_spformer import SPVCNN_SWIFTNET18_SPFORMER
    #     voxel_size = configs.dataset.voxel_size
    #     if not isinstance(voxel_size, list):
    #         voxel_size_list = [voxel_size] * 3
    #     else:
    #         voxel_size_list = voxel_size
    #     patch_size = np.array([voxel_size_list[i] * configs.model.patch_size for i in range(3)]).astype(
    #         np.float32)  # [0.1, 0.1, 0.1]
    #     window_size = patch_size * configs.model.window_size
    #     window_size_sphere = np.array(configs.model.window_size_sphere)
    #     model = SPVCNN_SWIFTNET18_SPFORMER(
    #         window_size=window_size,
    #         window_size_sphere=configs.model.window_size_sphere,
    #         quant_size=window_size / configs.model.quant_size_scale,
    #         quant_size_sphere=window_size_sphere / configs.model.quant_size_scale,
    #         drop_path_rate=configs.model.drop_path_rate,
    #         window_size_scale=configs.model.window_size_scale,
    #         a=configs.model.a,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size
    #     )
    # elif model_name == 'spvcnn_swiftnet18_spformer_msp2ifm':
    #     from core.models.nuscenes.spvcnn_swiftnet18_spformer_msp2ifm import SPVCNN_SWIFTNET18_SPFORMER_MSP2IFM
    #     voxel_size = configs.dataset.voxel_size
    #     if not isinstance(voxel_size, list):
    #         voxel_size_list = [voxel_size] * 3
    #     else:
    #         voxel_size_list = voxel_size
    #     patch_size = np.array([voxel_size_list[i] * configs.model.patch_size for i in range(3)]).astype(
    #         np.float32)  # [0.1, 0.1, 0.1]
    #     window_size = patch_size * configs.model.window_size
    #     window_size_sphere = np.array(configs.model.window_size_sphere)
    #     model = SPVCNN_SWIFTNET18_SPFORMER_MSP2IFM(
    #         window_size=window_size,
    #         window_size_sphere=configs.model.window_size_sphere,
    #         quant_size=window_size / configs.model.quant_size_scale,
    #         quant_size_sphere=window_size_sphere / configs.model.quant_size_scale,
    #         drop_path_rate=configs.model.drop_path_rate,
    #         window_size_scale=configs.model.window_size_scale,
    #         a=configs.model.a,
    #         pres=configs.dataset.voxel_size,
    #         vres=configs.dataset.voxel_size
    #     )
    elif model_name == 'spvcnn_swiftnet18_spformer_tsd_full':
        from core.models.nuscenes.spvcnn_swiftnet18_spformer_tsd_full import SPVCNN_SWIFTNET18_SPFORMER_TSD_FULL
        voxel_size = configs.dataset.voxel_size
        if not isinstance(voxel_size, list):
            voxel_size_list = [voxel_size] * 3
        else:
            voxel_size_list = voxel_size
        patch_size = np.array([voxel_size_list[i] * configs.model.patch_size for i in range(3)]).astype(
            np.float32)  # [0.1, 0.1, 0.1]
        window_size = patch_size * configs.model.window_size
        window_size_sphere = np.array(configs.model.window_size_sphere)
        model = SPVCNN_SWIFTNET18_SPFORMER_TSD_FULL(
            window_size=window_size,
            window_size_sphere=configs.model.window_size_sphere,
            quant_size=window_size / configs.model.quant_size_scale,
            quant_size_sphere=window_size_sphere / configs.model.quant_size_scale,
            drop_path_rate=configs.model.drop_path_rate,
            window_size_scale=configs.model.window_size_scale,
            a=configs.model.a,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size
        )
    else:
        raise NotImplementedError(model_name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lovasz':
        from core.criterions import MixLovaszCrossEntropy
        class_weight = configs['criterion'].get('class_weight', None)
        class_weight = torch.tensor(class_weight).cuda(non_blocking=True) if class_weight is not None else None
        criterion = MixLovaszCrossEntropy(weight=class_weight, ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lc_lovasz':
        from core.criterions import MixLCLovaszCrossEntropy
        criterion = MixLCLovaszCrossEntropy(ignore_index=configs.criterion.ignore_index)
    elif configs.criterion.name == 'lc_lovasz_distill':
        from core.criterions import DistillLovaszCrossEntropy
        criterion = DistillLovaszCrossEntropy(ignore_index=configs.criterion.ignore_index)
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_criterion_dict() -> dict[str, Callable]:
    ret_dict = {}
    for c_name in configs['criterion']['name']:
        if c_name == 'ce':
            ret_dict['ce'] = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)
        elif c_name == 'lovasz':
            from core.criterions import MixLovaszCrossEntropy
            ret_dict['lovasz'] = MixLovaszCrossEntropy(ignore_index=configs.criterion.ignore_index)
        elif c_name == 'kl':
            ret_dict['kl'] = nn.KLDivLoss(reduction='batchmean')
        elif c_name == 'mse':
            ret_dict['mse'] = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError(c_name)
    return ret_dict


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'sgd_spformer':
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if "transformer_block" not in n and p.requires_grad],
                "lr": configs.optimizer.lr,
                "momentum": configs.optimizer.momentum,
                "weight_decay": configs.optimizer.weight_decay,
                "nesterov": configs.optimizer.nesterov
            },
            {
                "params": [p for n, p in model.named_parameters() if "transformer_block" in n and p.requires_grad],
                "lr": configs.optimizer.lr * 0.1,
                "momentum": configs.optimizer.momentum,
                "weight_decay": configs.optimizer.weight_decay,
                "nesterov": configs.optimizer.nesterov
            },
        ]
        optimizer = torch.optim.SGD(
            param_dicts,
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw_spformer':
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if "transformer_block" not in n and p.requires_grad],
                "lr": configs.optimizer.lr,
                "weight_decay": configs.optimizer.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if "transformer_block" in n and p.requires_grad],
                "lr": configs.optimizer.lr * configs.optimizer.transformer_lr_scale,
                "weight_decay": configs.optimizer.weight_decay,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=configs.optimizer.lr, weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from core.schedulers import cosine_schedule_with_warmup
        from functools import partial
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                dataset_size=configs.data.training_size
            )
        )
    elif configs.scheduler.name == 'poly':
        from core.schedulers import PolyLR
        scheduler = PolyLR(
            optimizer, max_iter=configs.num_epochs * configs.data.training_size, power=configs.scheduler.power
        )
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
