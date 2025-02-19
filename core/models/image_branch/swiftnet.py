# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn
"""
    SwiftNet is a little different
    1. because it use the pre-activation input as lateral feature input.
    The backbone need writing for easier experiment
    2. I also add dsn head for easier training during the decoder upsample process.
    3. SwiftNet use torch pretrained backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo

from core.models.build_blocks import conv1x1, conv3x3, upsample
from typing import Type, Any, Callable, Union, List, Optional


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        efficient=True,
        use_bn=True
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.use_bn = use_bn

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        relu = self.relu(out)

        return relu, out


class SwiftNetResNet(nn.Module):
    def __init__(self, block, layers, num_features=(128, 128, 128), k_up=3, efficient=True, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, **kwargs):
        super(SwiftNetResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        # self.img_cs = [64, 64, 128, 256, 128]
        self.img_cs = [64] + [c * block.expansion for c in [64, 128, 256]] + [num_features[0]]
        self.im_cr = 1.0
        self.img_cs = [int(c * self.im_cr) for c in self.img_cs]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes_list = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes_list.append(self.inplanes)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes_list.append(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.inplanes_list.append(self.inplanes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        num_levels = 3
        self.spp_size = kwargs.get('spp_size', num_features)[0]
        bt_size = self.spp_size  # 128

        level_size = self.spp_size // num_levels  # 42

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=num_features[0], grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.024 / 2, use_bn=self.use_bn)

        self.build_decoder = kwargs.get('build_decoder', True)
        if self.build_decoder:
            upsamples = []
            upsamples += [
                _Upsample(num_features[1], self.inplanes_list[0], num_features[2], use_bn=self.use_bn, k=k_up)]
            upsamples += [
                _Upsample(num_features[0], self.inplanes_list[1], num_features[1], use_bn=self.use_bn, k=k_up)]
            upsamples += [
                _Upsample(num_features[0], self.inplanes_list[2], num_features[0], use_bn=self.use_bn, k=k_up)]
            self.upsample = nn.ModuleList(list(reversed(upsamples)))
        else:
            self.upsample = None

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        用于创建卷积层
        Args:
            block: <nn.Module> 用于构建网络的基本卷积模块，默认为标准的残差模块，即上面的BasicBlock
            planes: <int> 这个卷积层输出特征图的维度
            blocks: <int> 这个卷积层包含几个卷积模块
            stride: <int>

        Returns:

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_stem(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)  # skip=conv+bn+relu+conv+bn, x=relu(skip)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]

        x, skip = self.forward_resblock(x, self.layer4)

        features += [self.spp.forward(skip)]  # bn+relu+conv

        return features

    def forward_up(self, features, im_size=None):
        assert self.build_decoder
        features = features[::-1]

        x = features[0]  # spp.forward bn+relu+conv
        # upsamples = []
        for skip, up in zip(features[1:], self.upsample):  # conv+bn+relu
            x = up(x, skip)
            # upsamples += [x]
        # return upsamples[1:]
        if im_size is not None:
            return upsample(x, im_size)
        return x

    def forward(self, image, im_size=None):
        return self.forward_up(self.forward_down(image), im_size=im_size)


class SpatialPyramidPooling(nn.Module):
    """
        SPP module is little different from ppm by inserting middle level feature to save the computation and  memory.
    """

    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True):

        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids  # [8, 4, 2, 1]
        self.square_grid = square_grid  # default False
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]  # (H, W)

        ar = target_size[1] / target_size[0]  # W / H

        x = self.spp[0].forward(x)  # x=conv+bn+bn+relu+conv  512->128
        levels.append(x)
        num = len(self.spp) - 1  # 2

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))  # (grid[0], W/H * grid[0])
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(_Upsample, self).__init__()
        # print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        """
        x: bn+relu+conv
        skip: conv+bn
        """
        skip = self.bottleneck.forward(skip)  # conv+bn+bn+relu+conv
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)  # bn+relu+conv+bn+relu+conv
        return x


def SwiftNetRes18(num_feature=(128, 128, 128), **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        num_feature
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    build_decoder = kwargs.get('build_decoder', True)
    model = SwiftNetResNet(BasicBlock, [2, 2, 2, 2], num_feature, build_decoder=build_decoder)
    pretrained_path = kwargs.get('pretrained_path', None)
    if pretrained_path is not None:
        print("load pretrained weigth from", pretrained_path)
        model.load_state_dict(torch.load(pretrained_path), strict=False)
    else:
        print("train swiftnet from sketch")
    return model


def SwiftNetRes34(num_feature=(128, 128, 128), **kwargs):
    build_decoder = kwargs.get('build_decoder', True)
    model = SwiftNetResNet(BasicBlock, [3, 4, 6, 3], num_feature, build_decoder=build_decoder)
    pretrained_path = kwargs.get('pretrained_path', None)
    if pretrained_path is not None:
        print("load pretrained weigth from", pretrained_path)
        model.load_state_dict(torch.load(pretrained_path), strict=False)
    else:
        print("train swiftnet from sketch")
    return model


def SwiftNetRes50(num_feature=(128, 128, 128), **kwargs):
    build_decoder = kwargs.get('build_decoder', True)
    model = SwiftNetResNet(Bottleneck, [3, 4, 6, 3], num_feature, build_decoder=build_decoder)
    pretrained_path = kwargs.get('pretrained_path', None)
    if pretrained_path is not None:
        print("load pretrained weigth from", pretrained_path)
        model.load_state_dict(torch.load(pretrained_path), strict=False)
    else:
        print("train swiftnet from sketch")
    return model


if __name__ == '__main__':
    i = torch.Tensor(1, 3, 512, 512).cuda()
    weight_path = "/home/ubuntu/hdd1/stf2/codes/e3d/spvnas/pretrain/resnet18-5c106cde.pth"
    m = SwiftNetRes18(pretrained_path=weight_path).cuda()
    m.eval()
    o = m(i)
    print(o[0].size())
    print("output length: ", len(o))
