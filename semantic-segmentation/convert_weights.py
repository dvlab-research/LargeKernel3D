import os
import torch
import argparse

def _convert_list(kernel_size_src):
    _indice_list = []
    _list = [0, 3, 4, 7] if kernel_size_src == 7 else [0, 2, 3, 5]

    for i in range(len(_list) - 1):
        for j in range(len(_list) - 1):
            for k in range(len(_list) - 1):
                a = torch.zeros((kernel_size_src, kernel_size_src, kernel_size_src)).long()
                a[_list[i]:_list[i + 1], _list[j]:_list[j + 1], _list[k]:_list[k + 1]] = 1
                c = torch.range(0, kernel_size_src ** 3 - 1, 1)[a.reshape(-1).bool()]
                _indice_list.append(c.long())
    return _indice_list

def _convert_weight(weight, _indice_list, k_dst=3):
    k, _, _, out_channels, in_channels, = weight.shape
    weight_reshape = weight.permute(3, 4, 0, 1, 2).reshape(out_channels, in_channels, -1).clone()
    weight_return = torch.zeros(out_channels, in_channels, k_dst**3)
    for i, _indice in enumerate(_indice_list):
        _mean_weight = torch.mean(weight_reshape[:, :, _indice], dim=-1)
        weight_return[:, :, i] = _mean_weight
    return weight_return.reshape(out_channels, in_channels, k_dst, k_dst, k_dst).permute(2, 3, 4, 0, 1)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--checkpoint_path', type=str, default="/data/scannet_models/largekernel3d/model/model_last.pth")
    parser.add_argument('--output_path', type=str, default="/data/scannet_models/largekernel3d/model/")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_config()
    checkpoint_path = args.checkpoint_path
    output_path = args.output_path

    state_dict = {}
    checkpoint = torch.load(checkpoint_path)

    kernel_size_src = 7
    _indice_list = _convert_list(kernel_size_src)

    state_dict = {}
    for k in checkpoint['state_dict']:
        weight = checkpoint['state_dict'][k]
        weight_shape = weight.shape
        if len(weight_shape) == 5 and kernel_size_src in weight_shape:
            state_dict[k] = _convert_weight(weight, _indice_list, k_dst=3)
            print("Converted %s from %s to %s"%(k, str(weight_shape), str(state_dict[k].shape)))
        else:
            state_dict[k] = weight

    torch.save(state_dict, os.path.join(output_path, 'model_converted.pth'))