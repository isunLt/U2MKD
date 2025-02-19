from torchsparse import PointTensor

from core.models.utils import *
from core.models.build_blocks import *


__all__ = ['SPVCNN']


class SPVCNN(nn.Module):

    def __init__(self, **kwargs):
        super(SPVCNN, self).__init__()

        cr = kwargs.get('cr')
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        # cs = [32, 64, 128, 256, 256, 128, 128, 64, 32]
        cs = [int(cr * x) for x in cs]
        print('cr:', cr)
        print('cs:', cs)

        self.in_channel = kwargs.get('in_channel', 4)
        self.num_classes = kwargs.get('num_classes', 17)
        self.out_channel = cs[-1]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

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

        x0 = self.stem(x0)  # x0.F(N, 4) -> x0.F(N, 20)
        z0 = voxel_to_point(x0, z, nearest=False)

        vox_feats = [point_to_voxel(x0, z0)]
        for idx, (vox_block) in enumerate(self.vox_downs):
            vox_feats.append(vox_block(vox_feats[idx]))

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
