import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import SparseTensor
import torchsparse.nn as spnn


class IA_Layer(nn.Module):
    def __init__(self, channels, return_att=False):
        """
        ic: [64, 128, 256, 512]
        pc: [96, 256, 512, 1024]
        """
        super(IA_Layer, self).__init__()

        self.return_att = return_att
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU(True))
        # self.fc1 = nn.Linear(self.ic, rc)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.ic),
            nn.ReLU(True),
            nn.Linear(self.ic, rc)
        )
        self.fc2 = nn.Linear(self.pc, rc)
        # self.fc2 = nn.Sequential(
        #     nn.BatchNorm1d(self.pc),
        #     nn.ReLU(True),
        #     nn.Linear(self.pc, rc)
        # )
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feas: <Tensor, N, C> image_feature conv+bn
            point_feas: <Tensor, N, C'> point_feature conv+bn+relu

        Returns:

        """
        img_feats = img_feats.contiguous()
        point_feats = point_feats.contiguous()
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        att = att.unsqueeze(1).view(1, 1, -1)  # B1N

        img_feats_c = img_feats.unsqueeze(0).transpose(1, 2).contiguous()
        img_feas_new = self.conv1(img_feats_c)
        # 依据图像特征和点云特征的相关程度筛选图像特征
        out = img_feas_new * att

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes, return_att=False):
        """
        inplanes_I: [64, 128, 256, 512]
        inplanes_P: [96, 256, 512, 1024]
        outplanes: [96, 256, 512, 1024]
        """
        super(Atten_Fusion_Conv, self).__init__()

        self.return_att = return_att

        self.ai_layer = IA_Layer(channels=[inplanes_I, inplanes_P], return_att=return_att)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        """
        point_feature: 点云特征 [B, C, N] conv+bn+relu
        img_feature: 图像特征 [B, N, C]  conv+bn
        """

        img_features = self.ai_layer(img_features, point_features)  # [B, C, N]
        # print("img_features:", img_features.shape)
        point_feats = point_features.unsqueeze(0).transpose(1, 2)
        # 将筛选的图像特征与点云特征直接拼接
        fusion_features = torch.cat([point_feats, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        fusion_features = fusion_features.squeeze(0).transpose(0, 1)

        return fusion_features


class L2CAILayer(nn.Module):
    def __init__(self, channels):
        super(L2CAILayer, self).__init__()
        self.ic, self.pc = channels
        rc = self.ic // 4
        self.conv1 = nn.Sequential(nn.Conv2d(self.pc, self.ic, 1),
                                   nn.BatchNorm2d(self.ic),
                                   nn.ReLU(True))
        # self.fc1 = nn.Linear(self.ic, rc)
        self.fc1 = nn.Conv2d(self.ic, rc, kernel_size=1)
        self.fc2 = nn.Conv2d(self.pc, rc, kernel_size=1)
        self.fc3 = nn.Conv2d(rc, 1, kernel_size=1)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feas: <Tensor, N, C> image_feature conv+bn
            point_feas: <Tensor, N, C'> point_feature conv+bn+relu

        Returns:

        """
        img_feats = img_feats.contiguous()
        point_feats = point_feats.contiguous()
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        # att = att.unsqueeze(1).view(1, 1, -1)  # B1N

        # img_feats_c = img_feats.unsqueeze(0).transpose(1, 2).contiguous()
        # 依据图像特征和点云特征的相关程度筛选图像特征
        out = self.conv1(point_feats) * att

        return out


class L2CFusion(nn.Module):

    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(L2CFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes_I + inplanes_I, out_channels=outplanes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.l2c_ai_layer = L2CAILayer(channels=[inplanes_I, inplanes_P])

    def forward(self, point_features, img_features):
        """
        point_feature: 点云特征 [B, C, N] conv+bn+relu
        img_feature: 图像特征 [B, N, C]  conv+bn
        """

        l2c_features = self.l2c_ai_layer(img_features, point_features)  # [B, C, N]
        fusion_features = torch.cat([img_features, l2c_features], dim=1)
        fusion_features = self.bn1(self.conv1(fusion_features))

        return F.relu(fusion_features), fusion_features


class IL_FUSION_Layer(nn.Module):
    def __init__(self, channels):
        """
        ic: [64, 128, 256, 512]
        pc: [96, 256, 512, 1024]
        """
        super(IL_FUSION_Layer, self).__init__()

        self.ic, self.pc = channels
        rc = self.ic // 4
        self.conv1 = nn.Sequential(nn.Linear(self.pc, self.ic),
                                   nn.BatchNorm1d(self.ic),
                                   nn.ReLU(True))
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feas: <Tensor, N, C> image_feature conv+bn
            point_feas: <Tensor, N, C'> point_feature conv+bn+relu

        Returns:

        """
        img_feats = img_feats.contiguous()
        point_feats = point_feats.contiguous()
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats)
        rp = self.fc2(point_feats)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        sigma = torch.sigmoid(self.fc3(torch.tanh(ri + rp))).view(-1, 1)  # BNx1

        point_feats_new = self.conv1(point_feats)
        point_feats_new = point_feats_new * sigma
        return point_feats_new


class ILFusion(nn.Module):

    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(ILFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes_I + inplanes_I, out_channels=outplanes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        # self.l2c_ai_layer = IL_FUSION_Layer(channels=[inplanes_I, inplanes_P])

    def forward(self, point_features, img_features):
        """
        point_feature: 点云特征 [B, C, N] conv+bn+relu
        img_feature: 图像特征 [B, N, C]  conv+bn
        """
        # l2c_features = self.l2c_ai_layer(img_features, point_features)  # [B, C, N]
        fusion_features = torch.cat([img_features, point_features], dim=1)
        fusion_features = self.bn1(self.conv1(fusion_features))

        return F.relu(fusion_features), fusion_features



def Point2Grid(pts_feat, pixel_coordinates, masks, grid_size):
    cur = 0
    l2c_feat_map = []
    h, w = grid_size
    for mask, coord in zip(masks, pixel_coordinates):
        n = mask.size(1)
        bs_pts_feat = pts_feat[cur:cur + n, :]
        for co, ma in zip(coord, mask):
            u = (co[:, 0] + 1.0) / 2 * (w - 1.0)
            v = (co[:, 1] + 1.0) / 2 * (h - 1.0)
            uv = torch.floor(torch.stack([u, v], dim=1)).long()
            uv = torch.fliplr(uv[ma])
            uq, inv, count = torch.unique(uv, dim=0, return_inverse=True, return_counts=True)
            f2d = torch.zeros(size=(uq.size(0), pts_feat.F.size(1)), device=pts_feat.F.device)
            inv = inv.view(-1, 1).expand(-1, f2d.size(-1))
            f2d.scatter_add_(0, inv, bs_pts_feat[ma])
            f2d /= count.view(-1, 1)
            l2c_f = torch.sparse_coo_tensor(uq.transpose(0, 1).contiguous(), f2d, size=(h, w, f2d.size(-1)))
            l2c_feat_map.append(l2c_f.to_dense())
        cur += n
    l2c_feat_map = torch.stack(l2c_feat_map, dim=0).permute(0, 3, 1, 2).contiguous()
    return l2c_feat_map


def Feature_Gather(feature_map, xy, mode='bilinear'):
    """
    :param xy:(B,N,2), normalize to [-1,1], (width, height)
    :param feature_map:(B,C,H,W)
    :param mode: bilinear
    :return:
    """
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = nn.functional.grid_sample(feature_map, xy, padding_mode='zeros', align_corners=True,
                                                    mode=mode)  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def Feature_Fetch(masks, pix_coord, imfeats, mode='bilinear'):
    """

    Args:
        masks:
        pix_coord:
        imfeats: <Tensor, B, 6, C, H, W>
        mode: bilinear

    Returns:

    """
    imfs = []
    for mask, coord, img in zip(masks, pix_coord, imfeats):
        mask = mask.cuda()
        imf = torch.zeros(size=(mask.size(1), img.size(1))).cuda()
        imf_list = Feature_Gather(img, coord.cuda(), mode=mode).permute(0, 2, 1)  # [6, N, C]
        # assert mask.size(0) == coord.size(0)
        for idx in range(mask.size(0)):
            imf[mask[idx]] = imf_list[idx, mask[idx], :]
        imfs.append(imf)
    return torch.cat(imfs, dim=0)


"""
fusion block in CamLiFlow
"""

from third_party.csrc.wrapper import k_nearest_neighbor

class Conv1dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, act='leaky_relu'):
        super().__init__()

        self.conv_fn = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels, affine=True)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm == 'instance_norm_affine':
            self.norm_fn = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % act)

    def forward(self, x, return_skip=False):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        if return_skip:
            x_ = self.act_fn(x)
            return x, x_
        return self.act_fn(x)

class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, act='leaky_relu'):
        super().__init__()

        self.conv_fn = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm == 'instance_norm_affine':
            self.norm_fn = nn.InstanceNorm2d(out_channels, affine=True)
        # elif norm == 'layer_norm':
        #     self.norm_fn = LayerNormCF2d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % act)

    def forward(self, x, return_skip=False):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        if return_skip:
            x_ = self.act_fn(x)
            return x, x_
        return self.act_fn(x)

mesh_grid_cache = {}
def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]

@torch.cuda.amp.autocast(enabled=False)
def grid_sample_wrapper(feat_2d, uv):
    image_h, image_w = feat_2d.shape[2:]
    new_x = 2.0 * uv[:, 0] / (image_w - 1) - 1.0  # [bs, n_points]
    new_y = 2.0 * uv[:, 1] / (image_h - 1) - 1.0  # [bs, n_points]
    new_xy = torch.cat([new_x[:, :, None, None], new_y[:, :, None, None]], dim=-1)  # [bs, n_points, 1, 2]
    result = F.grid_sample(feat_2d.float(), new_xy, 'bilinear', align_corners=True)  # [bs, n_channels, n_points, 1]
    return result[..., 0]

def batch_indexing(batched_data: torch.Tensor, batched_indices: torch.Tensor, layout='channel_first'):
    def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, C, N]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, C, I1, I2, ..., Im]
        """
        def product(arr):
            p = 1
            for i in arr:
                p *= i
            return p
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size, n_channels = batched_data.shape[:2]
        indices_shape = list(batched_indices.shape[1:])
        batched_indices = batched_indices.reshape([batch_size, 1, -1])
        batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
        result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
        result = result.view([batch_size, n_channels] + indices_shape)
        return result

    def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
        """
        :param batched_data: [batch_size, N, C]
        :param batched_indices: [batch_size, I1, I2, ..., Im]
        :return: indexed data: [batch_size, I1, I2, ..., Im, C]
        """
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)  # [bs, I1, I2, ..., Im]
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == 'channel_first':
        return batch_indexing_channel_first(batched_data, batched_indices)
    elif layout == 'channel_last':
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise ValueError

class FusionAwareInterp(nn.Module):
    def __init__(self, n_channels_3d, k=1, norm=None):
        super().__init__()
        self.k = k
        self.out_conv = Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm)
        self.score_net = nn.Sequential(
            Conv2dNormRelu(3, 16),  # [dx, dy, |dx, dy|_2, sim]
            Conv2dNormRelu(16, n_channels_3d, act='sigmoid'),
        )

    def forward(self, uv, feat_2d, feat_3d):
        if feat_2d.dim() == 3:
            feat_2d = feat_2d.unsqueeze(0)
        if feat_3d.dim() == 2:
            feat_3d = feat_3d.transpose(0, 1).unsqueeze(0).contiguous()
        if uv.dim() == 2:
            uv = uv.transpose(0, 1).unsqueeze(0).contiguous()
        bs, _, image_h, image_w = feat_2d.shape
        n_channels_3d = feat_3d.shape[1]

        grid = mesh_grid(bs, image_h, image_w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        knn_indices = k_nearest_neighbor(uv, grid, self.k)  # [B, HW, k]

        knn_uv, knn_feat3d = torch.split(
            batch_indexing(
                torch.cat([uv, feat_3d], dim=1),
                knn_indices
            ), [2, n_channels_3d], dim=1)

        knn_offset = knn_uv - grid[..., None]  # [B, 2, HW, k]
        knn_offset_norm = torch.linalg.norm(knn_offset, dim=1, keepdim=True)  # [B, 1, HW, k]

        score_input = torch.cat([knn_offset, knn_offset_norm], dim=1)  # [B, 4, HW, K]
        score = self.score_net(score_input)  # [B, n_channels_3d, HW, k]
        # score = softmax(score, dim=-1)  # [B, n_channels_3d, HW, k]

        final = score * knn_feat3d  # [B, n_channels_3d, HW, k]
        final = final.sum(dim=-1).reshape(bs, -1, image_h, image_w)  # [B, n_channels_3d, H, W]
        final = self.out_conv(final)

        return final

class SKFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None, reduction=1):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError

        self.fc_mid = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_channels // reduction, out_channels * 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_2d, feat_3d, return_skip=False):
        bs = feat_2d.shape[0]

        if return_skip:
            skip_2d, feat_2d = self.align1(feat_2d, return_skip)
            skip_3d, feat_3d = self.align2(feat_3d, return_skip)
        else:
            feat_2d = self.align1(feat_2d, return_skip)
            feat_3d = self.align2(feat_3d, return_skip)

        weight = self.avg_pool(feat_2d + feat_3d).reshape([bs, -1])  # [bs, C]
        weight = self.fc_mid(weight)  # [bs, C / r]
        weight = self.fc_out(weight).reshape([bs, -1, 2])  # [bs, C, 2]
        weight = F.softmax(weight, dim=-1)
        w1, w2 = weight[..., 0], weight[..., 1]  # [bs, C]

        if len(feat_2d.shape) == 4:
            w1 = w1.reshape([bs, -1, 1, 1])
            w2 = w2.reshape([bs, -1, 1, 1])
        else:
            w1 = w1.reshape([bs, -1, 1])
            w2 = w2.reshape([bs, -1, 1])
        if return_skip:
            return feat_2d * w1 + feat_3d * w2, skip_2d * w1 + skip_3d * w1
        return feat_2d * w1 + feat_3d * w2

class CLFM(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn='sk', norm=None):
        super().__init__()

        self.interp = FusionAwareInterp(in_channels_3d, k=1, norm=norm)
        self.mlps3d = Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm)

        self.fuse2d = SKFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm, reduction=2)
        self.fuse3d = SKFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm, reduction=2)

    def forward(self, uv, feat_2d, feat_3d):
        feat_2d = feat_2d.float()
        feat_3d = feat_3d.float()

        feat_3d_interp = self.interp(uv, feat_2d.detach(), feat_3d.detach())
        out2d = self.fuse2d(feat_2d, feat_3d_interp)

        feat_2d_sampled = grid_sample_wrapper(feat_2d.detach(), uv)
        out3d = self.fuse3d(self.mlps3d(feat_2d_sampled.detach()), feat_3d)

        return out2d, out3d

"""
End for CamLiFlow
"""