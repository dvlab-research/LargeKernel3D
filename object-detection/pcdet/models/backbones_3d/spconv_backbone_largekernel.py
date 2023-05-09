from functools import partial
import torch
import torch.nn as nn
from spconv.core import ConvAlgo
from timm.models.layers import trunc_normal_

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SpatialGroupConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False):
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
                                    )

        self.conv3x3_1 = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=int(kernel_size//2)-1,
                                        bias=bias,
                                        dilation=int(kernel_size//2)-1,
                                        indice_key=indice_key+'conv_3x3_1',
                                    )
        self._indice_list = []

        if kernel_size==7:
            _list = [0, 3, 4, 7]
        elif kernel_size==5:
            _list = [0, 2, 3, 5]
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
            self.block.weight.data = self._convert_weight(self.block.weight.data)
        x_conv_block = self.block(x_conv)
        x_conv_conv3x3_1 = self.conv3x3_1(x_conv)
        x_conv_block = x_conv_block.replace_feature(x_conv_block.features + x_conv_conv3x3_1.features)
        return x_conv_block


class SpatialGroupConvV2(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, indice_key=None, position_embedding=True, use_relu=False):
        super(SpatialGroupConvV2, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size > 3, "SpatialGroupConv3d requires large kernels."
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

        self.block = spconv.SpatialGroupConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size, 3,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                        algo=ConvAlgo.Native,
                                        position_embedding=position_embedding,
                                        use_relu=use_relu,
                                    )
        if position_embedding:
            trunc_normal_(self.block.position_embedding, std=0.02)

    def forward(self, x_conv):
        x_conv = self.block(x_conv, group_map=self.group_map.to(x_conv.features.device))
        return x_conv


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, norm_fn=None, downsample=None, indice_key=None, conv_type='common'):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        if conv_type=="spatialgroupconv":
            conv_func = SpatialGroupConv
        elif conv_type == 'spatialgroupconvv2':
            conv_func = SpatialGroupConvV2
        elif conv_type=='common':
            conv_func = spconv.SubMConv3d
        else:
            raise ValueError('Unknown conv type %s.'%conv_type)

        self.conv1 = conv_func(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size//2), bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv_func(
            planes, planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size//2), bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelResBackBone8xLargeKernel3D(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        kernel_sizes = model_cfg.get('KERNEL_SIZES', [7, 5, 5, 3])
        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [5, 5])
        conv_types = model_cfg.get('CONV_TYPES', ['spatialgroupconv', 'common', 'common', 'common'])

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, kernel_sizes[0], norm_fn=norm_fn, indice_key='res1', conv_type=conv_types[0]),
            SparseBasicBlock(16, 16, kernel_sizes[0], norm_fn=norm_fn, indice_key='res1', conv_type=conv_types[0]),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, kernel_sizes[1], norm_fn=norm_fn, indice_key='res2', conv_type=conv_types[1]),
            SparseBasicBlock(32, 32, kernel_sizes[1], norm_fn=norm_fn, indice_key='res2', conv_type=conv_types[1]),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, kernel_sizes[2], norm_fn=norm_fn, indice_key='res3', conv_type=conv_types[2]),
            SparseBasicBlock(64, 64, kernel_sizes[2], norm_fn=norm_fn, indice_key='res3', conv_type=conv_types[2]),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, kernel_sizes[3], norm_fn=norm_fn, indice_key='res4', conv_type=conv_types[3]),
            SparseBasicBlock(128, 128, kernel_sizes[3], norm_fn=norm_fn, indice_key='res4', conv_type=conv_types[3]),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
