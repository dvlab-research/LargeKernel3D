import os
import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.common_util import read_txt
from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_scannet as data_prepare
from util.data_util import data_prepare_scannet_minkows as data_prepare_me
import glob
from plyfile import PlyData

class Scannetv2(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))
            
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]
        # print("idx:", idx)
        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        data = torch.load(data_path)
        
        coord, feat = data[0], data[1]
        # print("coord:", coord[0:10])
        # assert False
        if self.split != 'test':
            label = data[2]

        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label
        
    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list) * self.loop



class Scannetv2_ME(Dataset):
    NUM_LABELS=41
    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, teach_voxel_size=None, prevoxel_transforms=None, transform=None, ignore_label=-100, shuffle_index=False, return_ori_coords=False, distill_mode=None, loop=1): 
        super().__init__() 
        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.teach_voxel_size = teach_voxel_size
        # self.voxel_max = voxel_max
        self.prevoxel_transforms = prevoxel_transforms
        self.transform = transform
        self.return_ori_coords = return_ori_coords 
        self.ignore_label = ignore_label
        self.shuffle_index = shuffle_index
        self.distill_mode = distill_mode 
        self.loop = loop
        
        # map labels not evaluated to ignore_label
        label_map = {}
        n_used = 0
        for l in range(self.NUM_LABELS):
            if l in self.IGNORE_LABELS:
                label_map[l] = self.ignore_label
            else:
                label_map[l] = n_used
                n_used += 1
        label_map[self.ignore_label] = self.ignore_label
        self.label_map = label_map
        self.NUM_LABELS -= len(self.IGNORE_LABELS)
        
        DATA_PATH_FILE = {
            'train': 'scannetv2_train.txt',
            'val': 'scannetv2_val.txt',
            'trainval': 'scannetv2_trainval.txt',
            'test': 'scannetv2_test.txt'
        }
        self.data_list = read_txt(os.path.join(data_root, DATA_PATH_FILE[split]))
        
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))
    
    def load_ply(self, index):
        if self.split != 'test':
            phase = 'train'
        else:
            phase = 'test'

        filepath = os.path.join(self.data_root, phase, self.data_list[index])
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)
        return coords, feats, labels
    
    def __getitem__(self, idx):
        # data_idx = idx % len(self.data_list)
        # data_path = self.data_list[data_idx]
        # data = torch.load(data_path)

        # coord, feat = data[0], data[1]
        # print("check:", feat[0:10])
        # if self.split != 'test':
        #     label = data[2]
        coord, feat, label = self.load_ply(idx)
        # print('check coords:', coord[0:10])
        # print("feat max 0", feat.max(0), "coord max", coord.max(0))
        # Prevoxel transformations
        if self.prevoxel_transforms is not None:
            coord, feat, label = self.prevoxel_transforms(coord, feat, label)
        # coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        if self.teach_voxel_size:
            if self.return_ori_coords:
                coord, feat, label, ori_coord, coord_t, feat_t, label_t, ori_coord_t = data_prepare_me(coord, feat, label, self.voxel_size, teach_voxel_size=self.teach_voxel_size, return_ori_coords=True, use_aug=True if self.split=='train' else False)
            else:
                coord, feat, label, coord_t, feat_t, label_t = data_prepare_me(coord, feat, label, self.voxel_size, teach_voxel_size=self.teach_voxel_size, use_aug=True if self.split=='train' else False)

            # transform
            if self.transform is not None:
                if self.distill_mode is None:
                    if self.return_ori_coords:
                        coord, feat, label, ori_coord = self.transform(coord, feat, label, ori_coord)
                    else:
                        coord, feat, label = self.transform(coord, feat, label)
                elif self.return_ori_coords:
                    coord, feat, label, ori_coord, coord_t, feat_t, label_t, ori_coord_t = self.transform(coord, feat, label, ori_coord, coord_t, feat_t, label_t, ori_coord_t)
                else:
                    coord, feat, label, coord_t, feat_t, label_t = self.transform(coord, feat, label, coord_t, feat_t, label_t)
                # coord_t, feat_t, label_t = self.transform(coord_t, feat_t, label_t)
            # mask label
            if self.ignore_label is not None:
                label = np.array([self.label_map[x] for x in label], dtype=np.int)
                label_t = np.array([self.label_map[x] for x in label_t], dtype=np.int)
            if self.shuffle_index:
                shuf_idx1 = np.arange(coord.shape[0])
                shuf_idx2 = np.arange(coord_t.shape[0])
                np.random.shuffle(shuf_idx1)
                np.random.shuffle(shuf_idx2)
                coord, feat, label = coord[shuf_idx1], feat[shuf_idx1], label[shuf_idx1]
                coord_t, feat_t, label_t = coord_t[shuf_idx2], feat_t[shuf_idx2], label_t[shuf_idx2]
                if self.return_ori_coords:
                    ori_coord = ori_coord[shuf_idx1]
                    ori_coord_t = ori_coord_t[shuf_idx2]
                    # return tuple([coord, feat, label, ori_coord, coord_t, feat_t, label_t, ori_coord_t])
            if self.return_ori_coords:
                return tuple([coord, feat, label, ori_coord, coord_t, feat_t, label_t, ori_coord_t]) 
            else:
                return tuple([coord, feat, label, coord_t, feat_t, label_t])
        else:
            coord, feat, label = data_prepare_me(coord, feat, label, self.voxel_size, teach_voxel_size=self.teach_voxel_size, use_aug=True if self.split=='train' else False)

            # transform
            if self.transform is not None:
                coord, feat, label = self.transform(coord, feat, label)
            # mask label
            if self.ignore_label is not None:
                label = np.array([self.label_map[x] for x in label], dtype=np.int)

            if self.shuffle_index:
                shuf_idx = np.arange(coord.shape[0])
                np.random.shuffle(shuf_idx)
                coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

            return tuple([coord, feat, label])

    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list) * self.loop

        
        
