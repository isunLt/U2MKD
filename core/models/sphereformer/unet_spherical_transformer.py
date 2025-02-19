import functools
import warnings
import torch
import torch.nn as nn
import numpy as np
# import spconv.pytorch as spconv
# from spconv.pytorch.modules import SparseModule
# from spconv.core import ConvAlgo
import torchsparse.nn as spnn
from torchsparse import SparseTensor, PointTensor
from torchsparse.utils import make_ntuple
from core.models.utils import *

from collections import OrderedDict
from torch_scatter import scatter_mean
from core.models.sphereformer.spherical_transformer import SphereFormer


# class ResidualBlock(SparseModule):
#     def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
#         super().__init__()
#         if in_channels == out_channels:
#             self.i_branch = spconv.SparseSequential(
#                 nn.Identity()
#             )
#         else:
#             self.i_branch = spconv.SparseSequential(
#                 spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
#             )
#         self.conv_branch = spconv.SparseSequential(
#             norm_fn(in_channels),
#             nn.ReLU(),
#             spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
#             norm_fn(out_channels),
#             nn.ReLU(),
#             spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
#         )
#
#     def forward(self, input):
#         identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
#         output = self.conv_branch(input)
#         output = output.replace_feature(output.features + self.i_branch(identity).features)
#         return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = nn.Sequential(
                nn.Identity()
            )
        else:
            self.i_branch = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1, stride=1),
            )
        self.conv_branch = nn.Sequential(
            norm_fn(in_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(in_channels, out_channels, kernel_size=3, dilation=1, stride=1),
            norm_fn(out_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, stride=1),
        )

    def forward(self, x):
        x = self.conv_branch(x) + self.i_branch(x)
        return x


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn):
        super().__init__()
        self.conv_layers = nn.Sequential(
            norm_fn(in_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(in_channels, out_channels, kernel_size=3, dilation=1, stride=1),
        )

    def forward(self, input):
        return self.conv_layers(input)


def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = (pair_in != -1)
    valid_pair_in, valid_pair_out = pair_in[valid_mask].long(), pair_out[valid_mask].long()
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next


class UBlock(nn.Module):
    def __init__(self, nPlanes,  # [32, 64, 128, 256, 256]
                 norm_fn,  # nn.BatchNorm1d
                 block_reps,  # 2
                 block,  # ResidualBlock
                 window_size,  # patch_size * window_size = 1 * 6 = 6
                 window_size_sphere,  # [2, 2, 120]
                 quant_size,  # window_size / quant_size_scale = 4 / 24 = 1/6
                 quant_size_sphere,  # window_size_sphere / quant_size_scale = [2,2,120] / 24
                 head_dim=16,
                 window_size_scale=[2.0, 2.0],
                 rel_query=True,
                 rel_key=True,
                 rel_value=True,
                 drop_path=0.0,
                 indice_key_id=1,
                 grad_checkpoint_layers=[],
                 sphere_layers=[1, 2, 3, 4, 5],
                 a=0.05 * 0.25,
                 ):

        super().__init__()

        self.nPlanes = nPlanes  # [32, 64, 128, 256, 256]
        self.indice_key_id = indice_key_id  # 1
        self.grad_checkpoint_layers = grad_checkpoint_layers  # []
        self.sphere_layers = sphere_layers  # [1,2,3,4,5]

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn)
                  for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = nn.Sequential(blocks)

        if indice_key_id in sphere_layers:
            self.window_size = window_size
            self.window_size_sphere = window_size_sphere
            num_heads = nPlanes[0] // head_dim
            self.transformer_block = SphereFormer(
                nPlanes[0],
                num_heads,
                window_size,
                window_size_sphere,
                quant_size,
                quant_size_sphere,
                indice_key='sphereformer{}'.format(indice_key_id),
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[0],
                a=a,
            )

        if len(nPlanes) > 1:
            self.conv = nn.Sequential(
                norm_fn(nPlanes[0]),
                spnn.ReLU(inplace=True),
                spnn.Conv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, dilation=1)
            )

            window_size_scale_cubic, window_size_scale_sphere = window_size_scale
            window_size_next = np.array([
                window_size[0] * window_size_scale_cubic,  # 1.2
                window_size[1] * window_size_scale_cubic,  # 1.2
                window_size[2] * window_size_scale_cubic  # 1.2
            ])
            quant_size_next = np.array([
                quant_size[0] * window_size_scale_cubic,  # 0.05
                quant_size[1] * window_size_scale_cubic,  # 0.05
                quant_size[2] * window_size_scale_cubic  # 0.05
            ])
            window_size_sphere_next = np.array([
                window_size_sphere[0] * window_size_scale_sphere,  # 4
                window_size_sphere[1] * window_size_scale_sphere,  # 4
                window_size_sphere[2]  # 120
            ])
            quant_size_sphere_next = np.array([
                quant_size_sphere[0] * window_size_scale_sphere,  # 1/6
                quant_size_sphere[1] * window_size_scale_sphere,  # 1/6
                quant_size_sphere[2]  # 5
            ])
            self.u = UBlock(nPlanes[1:],
                            norm_fn,
                            block_reps,
                            block,
                            window_size_next,
                            window_size_sphere_next,
                            quant_size_next,
                            quant_size_sphere_next,
                            window_size_scale=window_size_scale,
                            rel_query=rel_query,
                            rel_key=rel_key,
                            rel_value=rel_value,
                            drop_path=drop_path[1:],
                            indice_key_id=indice_key_id + 1,
                            grad_checkpoint_layers=grad_checkpoint_layers,
                            sphere_layers=sphere_layers,
                            a=a
                            )

            self.deconv = nn.Sequential(
                norm_fn(nPlanes[1]),
                spnn.ReLU(inplace=True),
                spnn.Conv3d(nPlanes[1], nPlanes[0], kernel_size=2, stride=2, bias=False, transposed=True)
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn)
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = nn.Sequential(blocks_tail)

    def forward(self, inp, xyz, batch):

        assert (inp.C[:, 3] == batch).all()

        output = self.blocks(inp)  # resblock [N_s, 32] -> [N_s, 32]

        # transformer
        if self.indice_key_id in self.sphere_layers:
            if self.indice_key_id in self.grad_checkpoint_layers:
                def run(feats_, xyz_, batch_):
                    return self.transformer_block(feats_, xyz_, batch_)
                transformer_features = torch.utils.checkpoint.checkpoint(run, output.F, xyz, batch)
            else:
                transformer_features = self.transformer_block(output.F, xyz, batch)
            output.F = transformer_features

        identity = SparseTensor(feats=output.F, coords=output.C, stride=output.s)
        identity.cmaps = output.cmaps
        identity.kmaps = output.kmaps

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)

            indice_pairs = output_decoder.kmaps[(make_ntuple(output.s, ndim=3),
                                                 self.conv[2].kernel_size,
                                                 self.conv[2].stride,
                                                 make_ntuple(self.conv[2].dilation, ndim=3))][0]
            pair_in, pair_out = indice_pairs[:, 0], indice_pairs[:, 1]
            xyz_next = scatter_mean(xyz[pair_in], index=pair_out, dim=0)
            batch_next = scatter_mean(batch.float()[pair_in], index=pair_out, dim=0).long()

            # xyz_next = PointTensor(xyz, output.C.float())
            # xyz_next = point_to_voxel(output_decoder, xyz_next)
            # downsample
            # indice_pairs = output_decoder.indice_dict['spconv{}'.format(self.indice_key_id)].indice_pairs
            # xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)
            # xyz_next, batch_next = xyz_next.F, xyz_next.C[:, 3].long()

            output_decoder = self.u(output_decoder, xyz_next, batch_next.long())
            output_decoder = self.deconv(output_decoder)
            output.F = torch.cat((identity.F, output_decoder.F), dim=1)
            # output_x = SparseTensor(feats=torch.cat((identity.F, output_decoder.F), dim=1), coords=output.C, stride=output.s)
            # output_x.cmaps = output.cmaps
            # output_x.kmaps = output.kmaps
            output = self.blocks_tail(output)

        return output

    # def forward(self, inp, xyz, batch):
    #
    #     assert (inp.indices[:, 0] == batch).all()
    #
    #     output = self.blocks(inp)  # resblock [N_s, 32] -> [N_s, 32]
    #
    #     # transformer
    #     if self.indice_key_id in self.sphere_layers:
    #         if self.indice_key_id in self.grad_checkpoint_layers:
    #             def run(feats_, xyz_, batch_):
    #                 return self.transformer_block(feats_, xyz_, batch_)
    #
    #             transformer_features = torch.utils.checkpoint.checkpoint(run, output.features, xyz, batch)
    #         else:
    #             transformer_features = self.transformer_block(output.features, xyz, batch)
    #         output = output.replace_feature(transformer_features)
    #
    #     identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
    #
    #     if len(self.nPlanes) > 1:
    #         output_decoder = self.conv(output)
    #
    #         # downsample
    #         indice_pairs = output_decoder.indice_dict['spconv{}'.format(self.indice_key_id)].indice_pairs
    #         xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)
    #
    #         output_decoder = self.u(output_decoder, xyz_next, batch_next.long())
    #         output_decoder = self.deconv(output_decoder)
    #         output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
    #         output = self.blocks_tail(output)
    #
    #     return output


class Semantic(nn.Module):
    def __init__(self,
                 input_c,  # 4
                 m,  # 32
                 classes,  # 16
                 block_reps,  # 2
                 block_residual,  # True
                 layers,  # [32, 64, 128, 256, 256]
                 window_size,  # patch_size * window_size = 1 * 6 = 6
                 window_size_sphere,  # [2, 2, 120]
                 quant_size,  # window_size / quant_size_scale = 4 / 24 = 1/6
                 quant_size_sphere,  # window_size_sphere / quant_size_scale = [2,2,120] / 24
                 rel_query=True,
                 rel_key=True,
                 rel_value=True,
                 drop_path_rate=0.0,  # 0.3
                 window_size_scale=2.0,  # [2.0, 2.0]
                 grad_checkpoint_layers=[],
                 sphere_layers=[1, 2, 3, 4, 5],  # [1,2,3,4,5]
                 a=0.05 * 0.25,  # 0.0125
                 ):
        super().__init__()

        # norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        norm_fn = functools.partial(spnn.BatchNorm, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]

        #### backbone
        self.input_conv = nn.Sequential(
            spnn.Conv3d(input_c, m, kernel_size=3, stride=1)
        )

        self.unet = UBlock(layers,
                           norm_fn,
                           block_reps,
                           block,
                           window_size,
                           window_size_sphere,
                           quant_size,
                           quant_size_sphere,
                           window_size_scale=window_size_scale,
                           rel_query=rel_query,
                           rel_key=rel_key,
                           rel_value=rel_value,
                           drop_path=dpr,
                           indice_key_id=1,
                           grad_checkpoint_layers=grad_checkpoint_layers,
                           sphere_layers=sphere_layers,
                           a=a,
                           )

        self.output_layer = nn.Sequential(
            norm_fn(m),
            spnn.ReLU(inplace=True)
        )

        #### semantic segmentation
        self.linear = nn.Linear(m, classes)  # bias(default): True

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, in_mod: dict):
        '''
        :param input_map: (N), int, cuda
        '''
        pts = in_mod['lidar']
        z = PointTensor(pts.F, pts.C.float())
        x0 = initial_voxelize(z, 1.0, 1.0)
        output = self.input_conv(x0)  # <N_s,4> -> <N_s, 32>
        z0 = voxel_to_point(output, z, nearest=False)
        output = point_to_voxel(output, z0)
        output = self.unet(output, x0.F[:, :3], x0.C[:, 3])
        output = self.output_layer(output)

        #### semantic segmentation
        semantic_scores = self.linear(output.F)  # (N, nClass), float
        return {
            'x_vox': semantic_scores
        }

