import numpy as np
import random
import SharedArray as SA
import logging
import torch

from util.voxelize import voxelize, get_transformation_matrix
import MinkowskiEngine as ME

def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn_limit(batch, max_batch_points, logger):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    k = 0
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(coord[:k]), torch.cat(feat[:k]), torch.cat(label[:k]), torch.IntTensor(offset[:k])
    # return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

def collate_fn_partseg(batch):
    coord, feat, cls, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(cls), torch.cat(label), torch.IntTensor(offset)

class cfl_collate_fn_factory:
    """Generates collate function for coords, feats, labels.
        Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                    size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints
        
    def __call__(self, list_data):
        ori_flag = False
        if len(list(zip(*list_data))) == 3:
            muti_voxel_flag = False
            coords, feats, labels = list(zip(*list_data))
            coords_batch, feats_batch, labels_batch = [], [], []
        elif len(list(zip(*list_data))) == 6:
            muti_voxel_flag = True
            coords, feats, labels, coords_t, feats_t, labels_t = list(zip(*list_data))
            coords_batch, feats_batch, labels_batch, offset, coords_batch_t, feats_batch_t, labels_batch_t, offset_t = [], [], [], [], [], [], [], []
        else:
            muti_voxel_flag = True
            ori_flag = True
            coords, feats, labels, ori_coords, coords_t, feats_t, labels_t, ori_coords_t = list(zip(*list_data))
            coords_batch, feats_batch, labels_batch, ori_coords_batch, offset, coords_batch_t, feats_batch_t, labels_batch_t, ori_coords_batch_t, offset_t = [], [], [], [], [], [], [], [], [], []

        batch_id = 0
        batch_num_points = 0
        batch_num_points_t = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if muti_voxel_flag:
                num_points_t = coords_t[batch_id].shape[0]
                batch_num_points_t += num_points_t
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                    f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
                )
                break
            coords_batch.append(torch.from_numpy(coords[batch_id]).int())
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            # offset.append(batch_num_points)
            if muti_voxel_flag:
                offset.append(batch_num_points)
                coords_batch_t.append(torch.from_numpy(coords_t[batch_id]).int())
                feats_batch_t.append(torch.from_numpy(feats_t[batch_id]))
                labels_batch_t.append(torch.from_numpy(labels_t[batch_id]).int())
                offset_t.append(batch_num_points_t)
            if ori_flag:
                ori_coords_batch.append(torch.from_numpy(ori_coords[batch_id]))
                ori_coords_batch_t.append(torch.from_numpy(ori_coords_t[batch_id]))
            
            batch_id += 1
        # Concatenate all lists
        coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
        if muti_voxel_flag:
            coords_batch_t, feats_batch_t, labels_batch_t = ME.utils.sparse_collate(coords_batch_t, feats_batch_t, labels_batch_t)
            offset = torch.IntTensor(offset)
            offset_t = torch.IntTensor(offset_t)
            if ori_flag:
                return coords_batch, feats_batch.float(), labels_batch, torch.cat(ori_coords_batch), offset, coords_batch_t, feats_batch_t.float(), labels_batch_t, torch.cat(ori_coords_batch_t), offset_t
            
            return coords_batch, feats_batch.float(), labels_batch, offset, coords_batch_t, feats_batch_t.float(), labels_batch_t, offset_t
        
        return coords_batch, feats_batch.float(), labels_batch
    

def area_crop(coord, area_rate, split='train'):
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min; coord_max -= coord_min
    x_max, y_max = coord_max[0:2]
    x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
    if split == 'train' or split == 'trainval':
        x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
    else:
        x_s, y_s = (x_max - x_size) / 2, (y_max - y_size) / 2
    x_e, y_e = x_s + x_size, y_s + y_size
    crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
    return crop_idx


def load_kitti_data(data_path):
    data = np.fromfile(data_path, dtype=np.float32)
    data = data.reshape((-1, 4))  # xyz+remission
    return data


def load_kitti_label(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= (coord_min + coord_max) / 2.0
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v101(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_scannet(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        # print("check max befor coords:", coord.max())
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        # print("check max after coords:", coord.max())
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    # print("coord check:", coord[0:5], "coord max:", coord.max())
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label

def clip(coords, clip_bound, center=None, trans_aug_ratio=None):
    bound_min = np.min(coords, 0).astype(float)
    bound_max = np.max(coords, 0).astype(float)
    bound_size = bound_max - bound_min
    if center is None:
        center = bound_min + bound_size * 0.5
    if trans_aug_ratio is not None:
        trans = np.multiply(trans_aug_ratio, bound_size)
        center += trans
    lim = clip_bound

    if isinstance(clip_bound, (int, float)):
      if bound_size.max() < clip_bound:
        return None
      else:
        clip_inds = ((coords[:, 0] >= (-lim + center[0])) & \
            (coords[:, 0] < (lim + center[0])) & \
            (coords[:, 1] >= (-lim + center[1])) & \
            (coords[:, 1] < (lim + center[1])) & \
            (coords[:, 2] >= (-lim + center[2])) & \
            (coords[:, 2] < (lim + center[2])))
        return clip_inds

    # Clip points outside the limit
    clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) & \
        (coords[:, 0] < (lim[0][1] + center[0])) & \
        (coords[:, 1] >= (lim[1][0] + center[1])) & \
        (coords[:, 1] < (lim[1][1] + center[1])) & \
        (coords[:, 2] >= (lim[2][0] + center[2])) & \
        (coords[:, 2] < (lim[2][1] + center[2])))
    return clip_inds


def data_prepare_scannet_minkows(coord, feat, label, voxel_size=0.04, teach_voxel_size=None, use_aug=False, return_ori_coords=False):
    clip_bound = None
    scale_augmentation_bound = (0.9, 1,1)
    # ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
    rotation_augmentation_bound = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
    # TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
    translation_augmentation_ratio_bound = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    
    # clip bound
    if clip_bound is not None:
        trans_aug_ratio = np.zeros(3)
        for axis_ind, trans_ratio_bound in enumerate(translation_augmentation_ratio_bound):
            trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

        clip_inds = clip(coord, clip_bound, trans_aug_ratio=trans_aug_ratio)
        if clip_inds is not None:
            coord, feat = coord[clip_inds], feat[clip_inds]
        if label is not None:
            label = label[clip_inds]

    # get rotation and scale
    transform_set = get_transformation_matrix(voxel_size, scale_augmentation_bound, rotation_augmentation_bound, use_aug, teach_voxel_size)
    if teach_voxel_size:
        M_v_s, M_v_t, M_r = transform_set
    else:
        M_v_s, M_r = transform_set
    # M_v_s, M_r_s = get_transformation_matrix(voxel_size, scale_augmentation_bound, rotation_augmentation_bound, use_aug)
    
    # Apply transformations
    rigid_transformation_s = M_v_s
    if use_aug:
        rigid_transformation_s = M_r @ rigid_transformation_s

    homo_coords = np.hstack((coord, np.ones((coord.shape[0], 1), dtype=coord.dtype)))
    coords_aug_s_all = np.floor(homo_coords @ rigid_transformation_s.T[:, :3])

    coords_aug_s, feat_s, label_s = ME.utils.sparse_quantize(
        coords_aug_s_all, feat, labels=label)
    
    if return_ori_coords:
        # _, ori_coords_s, _ = ME.utils.sparse_quantize(
        #     coords_aug_s_all, coord, labels=label)
        ori_coords_s = coords_aug_s * voxel_size
    
    if teach_voxel_size:
        # M_v_t, M_r_t = get_transformation_matrix(teach_voxel_size, scale_augmentation_bound, rotation_augmentation_bound, use_aug)
        rigid_transformation_t = M_v_t
        if use_aug:
            rigid_transformation_t = M_r @ rigid_transformation_t

        # homo_coords_t = np.hstack((coord, np.ones((coord.shape[0], 1), dtype=coord.dtype)))
        coords_aug_t_all = np.floor(homo_coords @ rigid_transformation_t.T[:, :3])

        coords_aug_t, feat_t, label_t = ME.utils.sparse_quantize(
            coords_aug_t_all, feat, labels=label)

        if return_ori_coords:
            # _, ori_coords_t, _ = ME.utils.sparse_quantize(
            #     coords_aug_t_all, coord, labels=label)
            ori_coords_t = coords_aug_t * teach_voxel_size
            return coords_aug_s, feat_s, label_s, ori_coords_s, coords_aug_t, feat_t, label_t, ori_coords_t
        
        return coords_aug_s, feat_s, label_s, coords_aug_t, feat_t, label_t
        
    return coords_aug_s, feat_s, label_s
    

def data_prepare_v102(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    while voxel_max and label.shape[0] > voxel_max * 1.1:
        area_rate = voxel_max / float(label.shape[0])
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        x_max, y_max = coord_max[0:2]
        x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
        if split == 'train':
            x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
        else:
            x_s, y_s = 0, 0
        x_e, y_e = x_s + x_size, y_s + y_size
        crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
        if crop_idx.shape[0] < voxel_max // 8: continue
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v103(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 7
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v104(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 10
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v105(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord[:, 0:2] -= coord_min[0:2]
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
