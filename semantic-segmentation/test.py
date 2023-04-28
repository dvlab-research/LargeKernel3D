import os
import time
import random
import numpy as np
import logging
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnion
from util.common_util import read_txt
from plyfile import PlyData
from util.minkows_util import voxel_align_h_v2

import MinkowskiEngine as ME
import spconv.pytorch as spconv

random.seed(123)
np.random.seed(123)

SCANNET_COLOR_MAP_V2 = {
    0: [0, 0, 0],
    1: [230, 25, 75],
    2: [60, 180, 75],
    3: [255, 225, 25],
    4: [0, 130, 200],
    5: [245, 130, 48],
    6: [145, 30, 180],
    7: [70, 240, 240],
    8: [240, 50, 230],
    9: [210, 245, 60],
    10: [250, 190, 190],
    11: [0, 128, 128],
    12: [230, 190, 255],
    14: [170, 110, 40],
    16: [255, 250, 200],
    24: [128, 0, 0],
    28: [170, 255, 195],
    33: [128, 128, 0],
    34: [255, 215, 180],
    36: [0, 0, 128],
    39: [128, 128, 128]
}


NUM_LABELS=41
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
label_map = {}
n_used = 0
for l in range(NUM_LABELS):
    if l in IGNORE_LABELS:
        label_map[l] = -100
    else:
        label_map[l] = n_used
        n_used += 1
label_map[-100] = -100
NUM_LABELS -= len(IGNORE_LABELS)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification / Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    
    # get model
    if args.arch == "LargeKernel3D":
        from model_half import LargeKernel3D
        model = LargeKernel3D(3, 20)
    elif args.arch == "LargeKernel3DFirst4":
        from model import LargeKernel3DFirst4
        model = LargeKernel3DFirst4(3, 20)
    elif args.arch == "LargeKernel3DT":
        from model import LargeKernel3DT
        model = LargeKernel3DT(3, 20)
    elif args.arch == "LargeKernel3DFirst4T":
        from model import LargeKernel3DFirst4T
        model = LargeKernel3DFirst4T(3, 20)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    
    model = model.cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    logger.info('# Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    test(model, criterion, names)


def data_prepare():
    data_root = args.data_root
    data_list = read_txt(args.val_list)
    return data_root, data_list

def load_ply(data_root, file_name, voxel_size):
    filepath = os.path.join(data_root, 'train', file_name)
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)

    feats[:, :3] = feats[:, :3] / 255. - 0.5
    
    quantized_coords = np.floor(coords / voxel_size)
    quantized_coords = torch.from_numpy(quantized_coords).contiguous()

    inds, inverse_map = ME.utils.sparse_quantize(quantized_coords, return_index=True, return_maps_only=True, return_inverse=True)
    
    labels = np.array([label_map[x] for x in labels], dtype=np.int)

    quantized_coords, feats = quantized_coords[inds], feats[inds]

    batch = [(quantized_coords, feats)]

    coordinates_, featrues_ = list(zip(*batch))
    # print("coords shape:", coords.shape)
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)
    return coordinates, features, labels, coords, inverse_map

def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.eval()
    
    data_root, data_list = data_prepare()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    args.test_mode = args.get("test_mode", "low")
    device = torch.device('cuda')
    logger.info('test_mode: {}'.format(args.test_mode))
    time_set = []
    for file_name in data_list:
        # print('file_name:', file_name)
        logger.info('file_name: {}'.format(file_name))
        if args.test_mode == "low":
            coord, feat, label, ori_coords, inverse_map = load_ply(data_root, file_name, args.voxel_size)
        else:
            coord, feat, label, ori_coords, inverse_map = load_ply(data_root, file_name, args.voxel_size)
            coord_h, feat_h, _, ori_coords_h, inverse_map_h = load_ply(data_root, file_name, args.teach_voxel_size)
            

        start_time = time.time()
        with torch.no_grad():
            if args.test_mode != "low":
                offset = torch.IntTensor([[coord.shape[0]]]).cuda()
                offset_h = torch.IntTensor([[coord_h.shape[0]]]).cuda()
                
                ori_coords = torch.from_numpy(ori_coords).cuda()
                ori_coords_h = torch.from_numpy(ori_coords_h).cuda()
            #sinput = ME.SparseTensor(feat, coord, device='cuda')
            coord[:, 1:] -= coord[:, 1:].min(dim=0)[0].unsqueeze(0)
            sinput = spconv.SparseConvTensor(feat.to(device), coord.to(device), coord[:, 1:].max(dim=0)[0], args.batch_size)
            if args.test_mode == "high_me":
                soutput, smap = model(sinput, [coord_h, ori_coords, ori_coords_h, offset, offset_h])
            else:
                soutput, smap = model(sinput)
            soutput, smap = soutput.features, smap.features
            
            if args.test_mode == "high":
                # print('------------------true--------------------')
                soutput, smap = voxel_align_h_v2(ori_coords, ori_coords_h, smap, soutput, offset, offset_h)

                
        _, pred = soutput.max(1)
        pred = pred.cpu().numpy()

        # reproject 
        if args.test_mode == "low":
            pred = pred[inverse_map]
        else:
            pred = pred[inverse_map_h]
        end_time = time.time()
        time_set.append(end_time-start_time)

        intersection, union, target = intersectionAndUnion(pred, label, 20, -100)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        keep_idx = (label != args.ignore_label)
        colors_pred = np.array([SCANNET_COLOR_MAP_V2[VALID_CLASS_IDS[l]] for l in pred])
        ori_coords -= ori_coords.min(0)
        save_base_name = os.path.basename(file_name).split(".")[0]
        write_obj(ori_coords[keep_idx], colors_pred[keep_idx], os.path.join(args.save_folder, save_base_name)+'.obj')

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    # fps:
    time_sum = 0
    for time_b in time_set:
        time_sum += time_b
    logger.info('FPS: {}'.format(1.0/(time_sum/len(time_set))))
    # names = [line.rstrip('\n') for line in open('/home/jhliu/Research/MinkowskiEngine/examples/scannet_names.txt')]
    for i in range(20):
        print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

if __name__ == '__main__':
    main()
