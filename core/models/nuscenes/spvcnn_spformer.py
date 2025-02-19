from torchsparse import PointTensor
from torchsparse.utils import make_ntuple

from core.models.utils import *
from core.models.build_blocks import *
from torchpack.utils.config import configs
from core.models.sphereformer.spherical_transformer import SphereFormer

from torch_scatter import scatter_mean


__all__ = ['SPVCNN_SPFORMER']


class SPVCNN_SPFORMER(nn.Module):

    def __init__(self, window_size,
                 window_size_sphere,
                 quant_size, quant_size_sphere, window_size_scale, drop_path_rate, a, pres, vres):
        super(SPVCNN_SPFORMER, self).__init__()

        cr = configs['model']['cr']
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        # cs = [32, 64, 128, 256, 256, 128, 128, 64, 32]
        cs = [int(cr * x) for x in cs]
        print('cr:', cr)
        print('cs:', cs)

        self.in_channel = configs['model']['in_channel']
        self.num_classes = configs['data']['num_classes']
        self.out_channel = cs[-1]

        self.pres = pres
        self.vres = vres

        self.stem = nn.Sequential(
            spnn.Conv3d(self.in_channel, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.vox_downs = nn.ModuleList()
        for idx in range(4):
            down = nn.Sequential(
                BasicConvolutionBlock(cs[idx], cs[idx], ks=2, stride=2, dilation=1),
                ResidualBlock(cs[idx], cs[idx + 1], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[idx + 1], cs[idx + 1], ks=3, stride=1, dilation=1)
            )
            self.vox_downs.append(down)

        self.window_size = window_size
        self.window_size_sphere = window_size_sphere
        self.quant_size = quant_size
        self.quant_size_sphere = quant_size_sphere
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]
        head_dim = 16

        self.transformer_blocks = nn.ModuleList()
        for idx in range(1, 5):
            num_heads = cs[idx] // head_dim
            self.transformer_blocks.append(
                SphereFormer(
                    cs[idx],
                    num_heads,
                    self.window_size,
                    self.window_size_sphere,
                    self.quant_size,
                    self.quant_size_sphere,
                    indice_key='sphereformer{}'.format(idx+1),
                    rel_query=True,
                    rel_key=True,
                    rel_value=True,
                    drop_path=dpr[idx],
                    a=a
                )
            )
            window_size_scale_cubic, window_size_scale_sphere = window_size_scale
            self.window_size = self.window_size * window_size_scale_cubic
            self.quant_size = self.quant_size * window_size_scale_cubic
            self.window_size_sphere[0] = self.window_size_sphere[0] * window_size_scale_sphere
            self.window_size_sphere[1] = self.window_size_sphere[1] * window_size_scale_sphere
            self.quant_size_sphere[0] = self.quant_size_sphere[0] * window_size_scale_sphere
            self.quant_size_sphere[1] = self.quant_size_sphere[1] * window_size_scale_sphere

        self.vox_ups = nn.ModuleList()
        for idx in range(4, len(cs) - 1):
            up = nn.ModuleList([
                BasicDeconvolutionBlock(cs[idx], cs[idx + 1], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[idx + 1] + cs[len(cs) - 1 - (1 + idx)], cs[idx + 1], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[idx + 1], cs[idx + 1], ks=3, stride=1, dilation=1)
                )
            ])
            self.vox_ups.append(up)

        self.classifier_vox = nn.Sequential(nn.Linear(cs[8], self.num_classes))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in_mod):
        """
        x: SparseTensor 表示voxel
        z: PointTensor 表示point
        Args:
            x: SparseTensor, x.C:(u,v,w,batch_idx), x.F:(x,y,z,sig)
        Returns:
        """
        x = in_mod['lidar']
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)
        # coord_xyz, batch = x0.F[:, :3], x0.C[:, 3]
        zz = PointTensor(x0.F, x0.C.float())

        x0 = self.stem(x0)  # x0.F(N, 4) -> x0.F(N, 20)
        z0 = voxel_to_point(x0, z, nearest=False)  # z0->z

        vox_feats = [point_to_voxel(x0, z0)]
        for idx, (vox_block) in enumerate(self.vox_downs):
            vox_out = vox_block(vox_feats[idx])
            tmp_p = point_to_voxel(vox_out, zz)
            coord_xyz, batch = tmp_p.F[:, :3], tmp_p.C[:, 3]
            trans_feat = self.transformer_blocks[idx](vox_out.F, coord_xyz, batch)
            vox_out.F = trans_feat
            vox_feats.append(vox_out)

        x1 = vox_feats[1]
        x2 = vox_feats[2]
        x3 = vox_feats[3]
        x4 = vox_feats[4]
        z1 = voxel_to_point(x4, z0)

        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.vox_ups[0][0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.vox_ups[0][1](y1)

        y2 = self.vox_ups[1][0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.vox_ups[1][1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.vox_ups[2][0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.vox_ups[2][1](y3)

        y4 = self.vox_ups[3][0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.vox_ups[3][1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        vox_out = self.classifier_vox(z3.F)

        return {
            'x_vox': vox_out,
            # 'num_pts': [coord.size(1) for coord in in_mod['pixel_coordinates']]
        }
