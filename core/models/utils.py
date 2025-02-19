import torch
from torch import nn
import torchsparse.nn.functional as spf
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

from torchsparse.nn.utils import fapply
import torchsparse.nn as spnn

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point', 'fetch_idx']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)
    # 以网格坐标为键值，
    pc_hash = spf.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = spf.sphashquery(pc_hash, sparse_hash)  # source_hash, target_hash 返回pc_hash中每个元素在sparse_hash中的序号, 相当于unique中的inverse
    counts = spf.spcount(idx_query.int(), len(sparse_hash))  # 返回idx_query中每个序号有多少个元素, 相当于unique中的count

    inserted_coords = spf.spvoxelize(torch.floor(new_float_coord), idx_query, counts)  # 属于相同voxel中的点的特征求平均数作为这个voxel的特征
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = spf.spvoxelize(z.F, idx_query, counts)  # 属于相同voxel中的点的特征求平均数作为这个voxel的特征

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    # new_tensor.check()
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query  # z.feature中每个点属于哪个voxel
    z.additional_features['counts'][1] = counts  # 每个voxel有几个点
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get('idx_query') is None \
            or z.additional_features['idx_query'].get(x.s) is None:
        pc_hash = spf.sphash(
            torch.cat([
                # torch.floor(z.C[:, :3] / x.s).int() * x.s,
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],  # 若x.s[0]等于8, 则离中心voxel8×8×8内的所有voxel会被认为是属于同一个voxel
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = spf.sphash(x.C)
        idx_query = spf.sphashquery(pc_hash, sparse_hash)  # 找PointTensor中的点对应的SparseTensor中的voxel
        counts = spf.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = spf.spvoxelize(z.F, idx_query, counts)  # 计算属于同一个voxel的PointTensor中点的特征的均值作为这个voxel的特征
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps
    # new_tensor.coord_maps = x.coord_maps
    # new_tensor.kernel_maps = x.kernel_maps

    return new_tensor


# x: , z:
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    """

    Args:
        x: SparseTensor x.C:(N,4), x.F:(N,cs[0]*cr)
        z: PointTensor z.C:(N,4), x.F:(N,input_size)
        nearest: False

    Returns:

    """
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        # kr = KernelRegion(2, x.s, 1)
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)  # [K, 3]
        # off = kr.get_kernel_offset().to(z.F.device)
        old_hash = spf.sphash(
            torch.cat([
                # torch.floor(z.C[:, :3] / x.s).int() * x.s,
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)  # [K, N] 返回坐标+offset后的hash值
        pc_hash = spf.sphash(x.C.to(z.F.device))
        idx_query = spf.sphashquery(old_hash, pc_hash)
        weights = spf.calc_ti_weights(z.C, idx_query, scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = spf.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = spf.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


def fetch_idx(source_coords, target_coords):
    """
    返回source_coords中每个元素在target_coords中的序号
    Args:
        source_coords:
        target_coords:

    Returns:

    """
    assert isinstance(source_coords, torch.Tensor)
    assert isinstance(target_coords, torch.Tensor)
    src_hash = spf.sphash(source_coords)
    dst_hash = spf.sphash(target_coords)
    return spf.sphashquery(src_hash, dst_hash)


class SparseSyncBatchNorm(nn.SyncBatchNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`torch.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, spnn.BatchNorm):
            module_output = SparseSyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output

