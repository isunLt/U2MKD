from torchsparse import PointTensor
from torchsparse.utils import make_ntuple

from core.models.utils import *
from core.models.build_blocks import *
from torchpack.utils.config import configs
from core.models.sphereformer.spherical_transformer import SphereFormer
from core.models.image_branch.swiftnet import SwiftNetRes18, _BNReluConv
from core.models.fusion_blocks import Atten_Fusion_Conv, Feature_Gather, Feature_Fetch
from core.models.fusion_blocks import L2CFusion

from torch_scatter import scatter_mean


__all__ = ['SPVCNN_SWIFTNET18_SPFORMER_MSP2IFM']


class SPVCNN_SWIFTNET18_SPFORMER_MSP2IFM(nn.Module):

    def __init__(self, window_size,
                 window_size_sphere,
                 quant_size, quant_size_sphere, window_size_scale, drop_path_rate, a, pres, vres):
        super(SPVCNN_SWIFTNET18_SPFORMER_MSP2IFM, self).__init__()

        cr = configs['model']['cr']
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        # cs = [32, 64, 128, 256, 256, 128, 128, 64, 32]
        cs = [int(cr * x) for x in cs]
        print('cr:', cr)
        print('cs:', cs)

        imagenet_pretrain = configs['model']['imagenet_pretrain']
        build_decoder = configs['eval'].get('build_pix_decoder', True)
        self.pix_branch = SwiftNetRes18(num_feature=(128, 128, 128), pretrained_path=imagenet_pretrain, build_decoder=build_decoder)
        img_cs = self.pix_branch.img_cs

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

        self.c2l_fusion_blocks = nn.ModuleList()
        self.l2c_fusion_blocks = nn.ModuleList()
        for idx in range(1, 5):
            c2l_fusion_block = Atten_Fusion_Conv(inplanes_I=img_cs[idx], inplanes_P=cs[idx], outplanes=cs[idx])
            self.c2l_fusion_blocks.append(c2l_fusion_block)
            l2c_fusion_block = L2CFusion(inplanes_I=img_cs[idx], inplanes_P=cs[idx], outplanes=img_cs[idx])
            self.l2c_fusion_blocks.append(l2c_fusion_block)

        self.learner = nn.ModuleList()
        for idx in range(1, 5):
            self.learner.append(nn.Sequential(
                nn.Linear(cs[idx], img_cs[idx]),
                nn.BatchNorm1d(img_cs[idx]),
                nn.ReLU(True),
                nn.Linear(img_cs[idx], img_cs[idx]),
                nn.BatchNorm1d(img_cs[idx])
            ))

        if configs['model']['align_loss'] == 'mae':
            self.align_loss = nn.L1Loss()
        elif configs['model']['align_loss'] == 'mse':
            self.align_loss = nn.MSELoss()

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
        if build_decoder:
            self.classifier_pix = _BNReluConv(num_maps_in=self.pix_branch.num_features, num_maps_out=self.num_classes, k=1)

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
        self.run_pix_decoder = configs['eval']['run_pix_decoder']
        self.run_align_loss = configs['eval']['run_align_loss']

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
        im = in_mod['images']  # [B, 6, H, W, C]
        ib, _, ic, ih, iw = im.size()
        im = im.view(-1, ic, ih, iw)  # [B * 6, C, H, W]
        pixel_coordinates = in_mod['pixel_coordinates']
        masks = in_mod['masks']
        fov_mask = in_mod['fov_mask']
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)
        # coord_xyz, batch = x0.F[:, :3], x0.C[:, 3]
        zz = PointTensor(x0.F, x0.C.float())

        x0 = self.stem(x0)  # x0.F(N, 4) -> x0.F(N, 20)
        z0 = voxel_to_point(x0, z, nearest=False)  # z0->z

        # img_feats = self.pix_branch.forward_down(im)
        x_im = self.pix_branch.forward_stem(im)

        vox_feats = [point_to_voxel(x0, z0)]
        img_feats = []
        align_loss = []
        for idx, vox_block in enumerate(self.vox_downs):

            vox_out = vox_block(vox_feats[idx])
            tmp_p = point_to_voxel(vox_out, zz)
            coord_xyz, batch = tmp_p.F[:, :3], tmp_p.C[:, 3]
            trans_feat = self.transformer_blocks[idx](vox_out.F, coord_xyz, batch)
            vox_out.F = trans_feat
            pts_feat = voxel_to_point(vox_out, z0)

            x_im, skip = self.pix_branch.forward_resblock(x_im, getattr(self.pix_branch, 'layer%s' % str(idx+1)))
            if idx == (len(self.vox_downs) - 1):
                skip = self.pix_branch.spp.forward(skip)
            _, ifc, ifh, ifw = skip.size()

            cur = 0
            l2c_feat_map = []
            for mask, coord in zip(masks, pixel_coordinates):
                n = mask.size(1)
                bs_pts_feat = pts_feat.F[cur:cur+n, :]
                for co, ma in zip(coord, mask):
                    l2c_f = torch.zeros(size=(1, pts_feat.F.size(1), ifh, ifw), device=bs_pts_feat.device)
                    if torch.sum(ma) == 0:
                        l2c_feat_map.append(l2c_f / (len(self.vox_downs) - idx))
                        continue
                    cnt = 1
                    for _ in range(idx, len(self.vox_downs)):
                        c_ih = int(round(float(ifh) / cnt + 0.01))
                        c_iw = int(round(float(ifw) / cnt + 0.01))
                        u = (co[:, 0] + 1.0) / 2 * (c_iw - 1.0)
                        v = (co[:, 1] + 1.0) / 2 * (c_ih - 1.0)
                        uv = torch.floor(torch.stack([u, v], dim=1)).long()
                        uv = torch.fliplr(uv[ma])
                        uq, inv, count = torch.unique(uv, dim=0, return_inverse=True, return_counts=True)
                        f2d = torch.zeros(size=(uq.size(0), pts_feat.F.size(1)), device=pts_feat.F.device)
                        inv = inv.view(-1, 1).expand(-1, f2d.size(-1))
                        f2d.scatter_add_(0, inv, bs_pts_feat[ma])
                        f2d /= count.view(-1, 1)
                        tmp = torch.sparse_coo_tensor(
                            uq.transpose(0, 1).contiguous(), f2d, size=(c_ih, c_iw, f2d.size(-1))
                        ).to_dense().permute(2, 0, 1).contiguous().view(1, -1, c_ih, c_iw)
                        l2c_f += upsample(tmp, size=(ifh, ifw))
                        cnt *= 2
                    l2c_feat_map.append(l2c_f / (len(self.vox_downs) - idx))
                cur += n
            l2c_feat_map = torch.concat(l2c_feat_map, dim=0).contiguous()
            x_im, skip = self.l2c_fusion_blocks[idx](l2c_feat_map, skip)
            img_feats.append(skip)

            img_feat_tensor = []
            for mask, coord, img in zip(masks, pixel_coordinates, skip.view(ib, -1, ifc, ifh, ifw)):
                """
                    mask <Tensor, [6, N]>
                    coord <Tensor, [6, N, 2]>
                    img <Tensor, [6, C, H, W]>
                """
                imf = torch.zeros(size=(mask.size(1), img.size(1)), device=img.device)
                imf_list = Feature_Gather(img, coord).permute(0, 2, 1)  # [B, N, C]
                assert mask.size(0) == coord.size(0)
                for m_i in range(mask.size(0)):
                    imf[mask[m_i]] = imf_list[m_i, mask[m_i], :]
                img_feat_tensor.append(imf)
            img_feat_tensor = torch.cat(img_feat_tensor, dim=0)
            pseudo_img_feat = self.learner[idx](pts_feat.F)
            img_feat_tensor[~fov_mask] = pseudo_img_feat[~fov_mask]
            if self.run_align_loss:
                align_loss.append(self.align_loss(pseudo_img_feat[fov_mask], img_feat_tensor[fov_mask].detach()))
            # if self.run_align_loss:
            #     align_loss.append(self.align_loss(pseudo_img_feat[fov_mask], img_feat_tensor[fov_mask])) # for kitti
            # img_feat_tensor = pseudo_img_feat
            # pseudo_fused_feat = self.c2l_fusion_blocks[idx](pts_feat.F, pseudo_img_feat)
            pts_feat.F = self.c2l_fusion_blocks[idx](pts_feat.F, img_feat_tensor)
            vox_feats.append(point_to_voxel(vox_out, pts_feat))

        x1 = vox_feats[1]
        x2 = vox_feats[2]
        x3 = vox_feats[3]
        x4 = vox_feats[4]
        z1 = pts_feat

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

        ret_dict = {
            'x_vox': vox_out,
            'num_pts': [coord.size(1) for coord in in_mod['pixel_coordinates']],
            # 'mse_loss_fuse': mse_loss_fuse
        }

        if self.run_align_loss:
            ret_dict['align_loss'] = align_loss

        if self.run_pix_decoder:
            pix_upsamples = self.pix_branch.forward_up(img_feats, im_size=(ih, iw))
            fmap_pix = self.classifier_pix(pix_upsamples)
            fmap_pix = fmap_pix.view(ib, -1, fmap_pix.size(1), fmap_pix.size(2), fmap_pix.size(3))
            pix_out = Feature_Fetch(masks, pixel_coordinates, fmap_pix)
            ret_dict['x_pix'] = pix_out

        return ret_dict
