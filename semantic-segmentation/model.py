# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from timm.models.layers import trunc_normal_


class SpatialGroupConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, _type='A'):
        super(SpatialGroupConv, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                        #algo=ConvAlgo.Native
                                    )

        self.conv3x3_1 = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        bias=bias,
                                        dilation=3,
                                        indice_key=indice_key+'conv_3x3_1',
                                        #algo=ConvAlgo.Native
                                    )

        self._indice_list = []

        if kernel_size==7:
            _list = [0, 3, 4, 7]
        elif kernel_size==5:
            _list = [0, 2, 3, 5]
        elif kernel_size==3:
            _list = [0, 1, 2]
        else:
            raise ValueError('Unknown kernel size %d'%kernel_size)
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    b = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    self._indice_list.append(b.long())

    def _convert_weight(self, weight):
        weight_reshape = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        weight_return = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        for _indice in self._indice_list:
            _mean_weight = torch.mean(weight_reshape[:, :, _indice], dim=-1, keepdim=True)
            weight_return[:, :, _indice] = _mean_weight
        return weight_return.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size).permute(2, 3, 4, 0, 1)

    def forward(self, x_conv):
        if self.training:
            self.block.weight.data = self._convert_weight(self.block.weight.data).contiguous()
        x_conv_block = self.block(x_conv)

        x_conv_conv3x3_1 = self.conv3x3_1(x_conv)

        x_conv_block = x_conv_block.replace_feature(x_conv_block.features + x_conv_conv3x3_1.features)
        return x_conv_block


class SpatialGroupConvV2(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, dilation=1, _type='A'):
        super(SpatialGroupConvV2, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernel_size==3:
            kernel_size = 7
        _list = [0, int(kernel_size//2), int(kernel_size//2)+1, 7]
        self.group_map = torch.zeros((3**3, int(kernel_size//2)**3)) - 1
        _num = 0
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    _pos = a.sum()
                    self.group_map[_num][:_pos] = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    _num += 1
        self.group_map = self.group_map.int()
        position_embedding = True
        self.block = spconv.SpatialGroupConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size, 3,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        dilation=dilation,
                                        indice_key=indice_key,
                                        algo=ConvAlgo.Native,
                                        position_embedding=position_embedding,
                                    )
        if position_embedding:
            trunc_normal_(self.block.position_embedding, std=0.02)

    def forward(self, x_conv):
        x_conv = self.block(x_conv, group_map=self.group_map.to(x_conv.features.device))
        return x_conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 norm_func=nn.BatchNorm1d,
                 act_func=nn.ReLU,
                 bn_momentum=0.1,
                 dimension=-1,
                 conv_type="common",
                 indice_key=''):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        if conv_type=="spatialgroupconv":
            conv_func = SpatialGroupConv
        elif conv_type == 'spatialgroupconvv2':
            conv_func = SpatialGroupConvV2
        elif conv_type =='common':
            conv_func = spconv.SubMConv3d
        else:
            raise ValueError('Unknown conv_type %s.' % conv_type)

        self.conv1 = conv_func(inplanes, planes, kernel_size=kernel_size[0], stride=stride,
                        padding=int(kernel_size[0]//2), bias=False, indice_key=indice_key+'conv1')
        self.norm1 = norm_func(planes, momentum=bn_momentum) if norm_func == nn.BatchNorm1d else norm_func(planes)
        self.conv2 = conv_func(planes, planes, kernel_size=kernel_size[1], stride=1,
                        padding=int(kernel_size[1]//2), bias=False, indice_key=indice_key+'conv2')
        self.norm2 = norm_func(planes, momentum=bn_momentum) if norm_func == nn.BatchNorm1d else norm_func(planes)
        self.relu = act_func()

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.norm1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))

        if self.downsample is not None:
          residual = self.downsample(x)

        out = out.replace_feature(out.features + residual.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class MinkUNetBaseSpconv(nn.Module):
    BLOCK = None
    PLANES = None
    KERNEL_SIZES = (3, 3, 3, 3, 3, 3, 3, 3)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1
    CONCATEXYZ = False
    VIRTUAL_VOXELS = False
    NORM_FUNC = nn.BatchNorm1d
    ACT_FUNC = nn.ReLU
    CONV_TYPE = ('common', 'common', 'common', 'common',
                 'common', 'common', 'common', 'common')

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, self.NORM_FUNC):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1, dilation=1, bn_momentum=0.1,
                        indice_key='', conv_type='common', norm_func=nn.BatchNorm1d, act_func=nn.ReLU):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = spconv.SparseSequential(
                spconv.SubMConv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                        bias=False, indice_key=indice_key+'downsample'),
                norm_func(planes * block.expansion, momentum=bn_momentum) if norm_func == nn.BatchNorm1d else norm_func(planes * block.expansion)
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
                conv_type=conv_type,
                norm_func=norm_func,
                act_func=act_func,
                indice_key=indice_key
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, kernel_size=kernel_size, stride=1, dilation=dilation, dimension=self.D,
                    conv_type=conv_type, norm_func=norm_func, act_func=act_func, indice_key=indice_key + str(i)
                )
            )
        return nn.Sequential(*layers)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = spconv.SubMConv3d(in_channels, self.inplanes, kernel_size=5, stride=1,
                             padding=2, bias=False, indice_key='conv0p1s1')

        self.bn0 = self.NORM_FUNC(self.inplanes)

        self.conv1p1s2 = spconv.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2,
                             padding=1, bias=False, indice_key='conv1p1s2', algo=ConvAlgo.Native)

        self.bn1 = self.NORM_FUNC(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], dilation=self.DILATIONS[0],
                        kernel_size=self.KERNEL_SIZES[0], indice_key='conv1p1s2_block', conv_type=self.CONV_TYPE[0],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)

        self.conv2p2s2 = spconv.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2,
                             padding=1, bias=False, indice_key='conv2p2s2', algo=ConvAlgo.Native)

        self.bn2 = self.NORM_FUNC(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], dilation=self.DILATIONS[1],
                        kernel_size=self.KERNEL_SIZES[1], indice_key='conv2p2s2_block', conv_type=self.CONV_TYPE[1],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)

        self.conv3p4s2 = spconv.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2,
                             padding=1, bias=False, indice_key='conv3p4s2', algo=ConvAlgo.Native)

        self.bn3 = self.NORM_FUNC(self.inplanes)

        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], dilation=self.DILATIONS[2],
                        kernel_size=self.KERNEL_SIZES[2], indice_key='conv3p4s2_block', conv_type=self.CONV_TYPE[2],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)

        self.conv4p8s2 = spconv.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2,
                             padding=1, bias=False, indice_key='conv4p8s2', algo=ConvAlgo.Native)

        self.bn4 = self.NORM_FUNC(self.inplanes)

        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], dilation=self.DILATIONS[3],
                        kernel_size=self.KERNEL_SIZES[3], indice_key='conv4p8s2_block', conv_type=self.CONV_TYPE[3],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)

        self.convtr4p16s2 = spconv.SparseInverseConv3d(self.inplanes, self.PLANES[4], kernel_size=2, 
                                        indice_key='conv4p8s2', bias=False, algo=ConvAlgo.Native)

        self.bntr4 = self.NORM_FUNC(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4], dilation=self.DILATIONS[4],
                        kernel_size=self.KERNEL_SIZES[4], indice_key='convtr4p16s2_block', conv_type=self.CONV_TYPE[4],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)

        self.convtr5p8s2 = spconv.SparseInverseConv3d(self.inplanes, self.PLANES[5], kernel_size=2, 
                                        indice_key='conv3p4s2', bias=False, algo=ConvAlgo.Native)

        self.bntr5 = self.NORM_FUNC(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5], dilation=self.DILATIONS[5],
                        kernel_size=self.KERNEL_SIZES[5], indice_key='convtr5p8s2_block', conv_type=self.CONV_TYPE[5],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)

        self.convtr6p4s2 = spconv.SparseInverseConv3d(self.inplanes, self.PLANES[6], kernel_size=2, 
                                        indice_key='conv2p2s2', bias=False, algo=ConvAlgo.Native)

        self.bntr6 = self.NORM_FUNC(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6], dilation=self.DILATIONS[6],
                        kernel_size=self.KERNEL_SIZES[6], indice_key='convtr6p4s2_block', conv_type=self.CONV_TYPE[6],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)

        self.convtr7p2s2 = spconv.SparseInverseConv3d(self.inplanes, self.PLANES[7], kernel_size=2, 
                                indice_key='conv1p1s2', bias=False, algo=ConvAlgo.Native)

        self.bntr7 = self.NORM_FUNC(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7], dilation=self.DILATIONS[7],
                        kernel_size=self.KERNEL_SIZES[7], indice_key='convtr7p2s2_block', conv_type=self.CONV_TYPE[7],
                        norm_func=self.NORM_FUNC, act_func=self.ACT_FUNC)
        
        if self.CONCATEXYZ:
            final_in_channel = (self.PLANES[7]+3) * self.BLOCK.expansion 
        else:
            final_in_channel = self.PLANES[7] * self.BLOCK.expansion 

        self.final = spconv.SubMConv3d(final_in_channel, out_channels, kernel_size=1, bias=True, indice_key='final')
        self.relu = nn.ReLU(inplace=True)

    def _recover_voxels(self, x, indices_ori):
        x = x.replace_feature(x.features[:indices_ori.shape[0]])
        x.indices = x.indices[:indices_ori.shape[0]]
        return x

    def forward(self, x):

        x1 = x
        out = self.conv0p1s1(x1)

        out = out.replace_feature(self.bn0(out.features))
        out_p1 = out.replace_feature(self.relu(out.features))

        out = self.conv1p1s2(out_p1)

        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features))

        out_b2p4 = self.block2(out)
        out = self.conv3p4s2(out_b2p4)

        out = out.replace_feature(self.bn3(out.features))
        out = out.replace_feature(self.relu(out.features))

        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)

        out = out.replace_feature(self.bn4(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.block4(out)

        out = self.convtr4p16s2(out)

        out = out.replace_feature(self.bntr4(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = out.replace_feature(torch.cat((out.features, out_b3p8.features), dim=1))

        out = self.block5(out)

        out = self.convtr5p8s2(out)

        out = out.replace_feature(self.bntr5(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = out.replace_feature(torch.cat((out.features, out_b2p4.features), dim=1))

        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = out.replace_feature(self.bntr6(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = out.replace_feature(torch.cat((out.features, out_b1p2.features), dim=1))

        out = self.block7(out)

        out = self.convtr7p2s2(out)

        out = out.replace_feature(self.bntr7(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = out.replace_feature(torch.cat((out.features, out_p1.features), dim=1))

        out = self.block8(out)

        return [self.final(out), out]



class LargeKernel3D(MinkUNetBaseSpconv):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    KERNEL_SIZES = ([7, 7], [7, 7], [7, 7], [7, 7], [3, 3], [3, 3], [3, 3], [3, 3])
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    CONV_TYPE = ('spatialgroupconv', 'spatialgroupconv', 'spatialgroupconv', 'spatialgroupconv',
                 'common', 'common', 'common', 'common')
