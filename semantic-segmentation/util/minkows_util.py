import os
import argparse
import numpy as np
from urllib.request import urlretrieve
try:
  import open3d as o3d
except ImportError:
  raise ImportError('Please install open3d with `pip install open3d`.')

import torch
import MinkowskiEngine as ME

#from lib.pointops2.functions.pointops import knnquery
#from lib.pointops2.functions import pointops
#from lib.pointgroup_ops.functions import pointgroup_ops

def pre_voxelize(coords, feats, voxel_size):

    quantized_coords = np.floor(coords / voxel_size)
    inds, inverse_map = ME.utils.sparse_quantize(quantized_coords, return_index=True, return_inverse=True)
    # print("inverse:", inverse.max(), "inds shape:", inds.shape)
    return quantized_coords[inds], feats[inds], inverse_map

def generate_input_sparse_tensor(coord, feat, offset, voxel_size=0.02):
    # Create a batch, this process is done in a data loader during training in parallel.
    # batch = [load_file(file_name, voxel_size)]
    # quantized_coords, feats, pcd, coords, inverse_map = pre_voxelize(file_name, voxel_size)
    # print("coord shape:", coord.shape, "feat shape:", feat.shape, "offset:", offset)
    batch = []
    inverse_maps = []
    inverse_offset = []
    count = 0
    for i in range(offset.shape[0]):
        if i == 0:
            coord_, feat_ = coord[0:offset[0]], feat[0:offset[0]]
            # batch.append((coord[0:offset[0]], feat[0:offset[0]]))
        else:
            coord_, feat_ = coord[offset[i-1]:offset[i]], feat[offset[i-1]:offset[i]]
            # batch.append((coord[offset[i-1]:offset[i]], feat[offset[i-1]:offset[i]]))
        coord_, feat_, inverse_map = pre_voxelize(coord_, feat_, voxel_size)
        batch.append((coord_, feat_))
        inverse_maps.append(inverse_map)
        count += coord_.shape[0]
        inverse_offset.append(count)
        # if i == 0:
        #     batch.append((coord[0:offset[0]], feat[0:offset[0]]))
        # else:
        #     batch.append((coord[offset[i-1]:offset[i]], feat[offset[i-1]:offset[i]]))
    # batch = [(quantized_coords, feats, pcd)]
    coordinates_, featrues_ = list(zip(*batch))
    # print("len coordinates_:", len(coordinates_))
    # print("coords shape:", coords.shape)
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)
    # print("coordinates shape:", coordinates.shape, "features shape:", features.shape)
    # Normalize features and create a sparse tensor
    return coordinates, (features - 0.5).float(), inverse_maps, inverse_offset

def voxel_align_l(nsample, stu_input, teach_input, teach_map, teach_output, offset, offset_t, stu_voxel_size=0.05, teach_voxel_size=0.02, shift_size=None):
    # stu_input.C: (m, 4), teach_input.C: (n, 4)
    if shift_size is not None:
        shift_size = shift_size.to(teach_input.C.device)
        teach_coords = (teach_input.C[:, 1:].float().contiguous() - shift_size) * teach_voxel_size
        stu_coords = (stu_input.C[:, 1:].float().contiguous() - shift_size) * stu_voxel_size
    else:
        teach_coords = (teach_input.C[:, 1:].float().contiguous()) * teach_voxel_size
        stu_coords = (stu_input.C[:, 1:].float().contiguous()) * stu_voxel_size
    
    idx, _ = knnquery(nsample, teach_coords, stu_coords, offset_t, offset) # [m, nsample]
    teach_map = teach_map[idx.long()].mean(1)
    teach_output = teach_output[idx.long()].mean(1)
    # print("idx shape: ", idx.shape, "teach_map shape: ",teach_map.shape, "teach_output shape: ", teach_output.shape)
    return teach_output, teach_map

def voxel_align_l_v2(nsample, stu_coords, teach_coords, teach_map, teach_output, offset, offset_t):
    # stu_input.C: (m, 4), teach_input.C: (n, 4)

    stu_coords = stu_coords.float().contiguous()
    teach_coords = teach_coords.float().contiguous()
        
    idx, _ = knnquery(nsample, teach_coords, stu_coords, offset_t, offset) # [m, nsample]
    if teach_map is not None:
        teach_map = teach_map[idx.long()].mean(1)
    if teach_output is not None:
        teach_output = teach_output[idx.long()].mean(1)
    # print("idx shape: ", idx.shape, "teach_map shape: ",teach_map.shape, "teach_output shape: ", teach_output.shape)
    return teach_output, teach_map

def voxel_align_h(stu_input, teach_input, stu_map, stu_output, offset, offset_t, stu_voxel_size=0.05, teach_voxel_size=0.02, shift_size=None):
    # stu_input.C: (m, 4), teach_input.C: (n, 4)
    # idx, _ = knnquery(nsample, teach_input.C[:, 1:].float().contiguous(), stu_input.C[:, 1:].float().contiguous(), offset_t, offset) # [m, nsample]
    # teach_map = teach_map[idx.long()].mean(1)
    # teach_output = teach_output[idx.long()].mean(1)
    # print("idx shape: ", idx.shape, "teach_map shape: ",teach_map.shape, "teach_output shape: ", teach_output.shape)

    if shift_size is not None:
        shift_size = shift_size.to(teach_input.C.device)
        teach_coords = (teach_input.C[:, 1:].float().contiguous() - shift_size) * teach_voxel_size
        stu_coords = (stu_input.C[:, 1:].float().contiguous() - shift_size) * stu_voxel_size
    else:
        teach_coords = (teach_input.C[:, 1:].float().contiguous()) * teach_voxel_size
        stu_coords = (stu_input.C[:, 1:].float().contiguous()) * stu_voxel_size

    if stu_map is not None:
        stu_map = pointops.interpolation(stu_coords, teach_coords, stu_map, offset, offset_t)
    stu_output = pointops.interpolation(stu_coords, teach_coords, stu_output, offset, offset_t)
    
    return stu_output, stu_map

def voxel_align_h_v2(stu_coords, teach_coords, stu_map, stu_output, offset, offset_t):
    # stu_input.C: (m, 4), teach_input.C: (n, 4)
    # idx, _ = knnquery(nsample, teach_input.C[:, 1:].float().contiguous(), stu_input.C[:, 1:].float().contiguous(), offset_t, offset) # [m, nsample]
    # teach_map = teach_map[idx.long()].mean(1)
    # teach_output = teach_output[idx.long()].mean(1)
    # print("idx shape: ", idx.shape, "teach_map shape: ",teach_map.shape, "teach_output shape: ", teach_output.shape)

    # if shift_size is not None:
    #     shift_size = shift_size.to(teach_input.C.device)
    #     teach_coords = (teach_input.C[:, 1:].float().contiguous() - shift_size) * teach_voxel_size
    #     stu_coords = (stu_input.C[:, 1:].float().contiguous() - shift_size) * stu_voxel_size
    # else:
    #     teach_coords = (teach_input.C[:, 1:].float().contiguous()) * teach_voxel_size
    #     stu_coords = (stu_input.C[:, 1:].float().contiguous()) * stu_voxel_size
    stu_coords = stu_coords.float().contiguous()
    teach_coords = teach_coords.float().contiguous()
    
    if stu_map is not None:
        stu_map = pointops.interpolation(stu_coords, teach_coords, stu_map, offset, offset_t)
    if stu_output is not None:
        stu_output = pointops.interpolation(stu_coords, teach_coords, stu_output, offset, offset_t)
    
    return stu_output, stu_map

def SparseTensorAdapt(x, adapt_items, concate_xyz=False):
    coord_t, ori_coords, ori_coords_t, offset, offset_t = adapt_items
    x_feats = x.F
    x_feats, _ = voxel_align_h_v2(ori_coords, ori_coords_t, None, x_feats, offset, offset_t)
    if concate_xyz:
        x_feats = torch.cat([x_feats, ori_coords_t], dim=1).float()
    x = ME.SparseTensor(x_feats, coord_t, device=x_feats.device)
    return x

def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    # print("max idx:", batch_idxs.max(), "bs:",bs,shape)
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    # print("batch_offsets[-1]:", batch_offsets[-1], "batch_idxs:", batch_idxs)
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets



def BFSClusterlabel(coords, label, batch_idxs, cluster_radius=0.03, cluster_meanActive=50, cluster_npoint_thre=50):
    
    object_idxs = torch.nonzero(label > 1).view(-1)
    masked_labels = label[object_idxs].int().cpu()
    coords = coords[object_idxs]
    batch_idxs = batch_idxs[object_idxs]
    batch_offsets = get_batch_offsets(batch_idxs, 1)
    
    # print("coords shape:", coords.shape)
    
    coords = coords.contiguous()
    idx, start_len = pointgroup_ops.ballquery_batch_p(coords, batch_idxs, batch_offsets, cluster_radius, cluster_meanActive)
    # print("idx:", idx)
    # assert False
    proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(masked_labels, idx.cpu(), start_len.cpu(), cluster_npoint_thre)
    # proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
    
    # print("proposals_idx:", proposals_idx)
    # assert False
    
    c_idxs = proposals_idx[:, 1].cuda()
    # print("c_idxs:", c_idxs.max())
    # assert False
    # print("coords:", coords.shape)
    # assert False
    clusters_coords = coords[c_idxs.long()]
    
    # assert False
    
    cluster_label = proposals_idx[:, 0]
    
    return clusters_coords, cluster_label



def BFSCluster(coords, label, batch_idxs, batch_size, cluster_radius=0.03, cluster_meanActive=50, cluster_npoint_thre=50):
    assert batch_idxs.shape[0] == coords.shape[0] == label.shape[0]
    object_idxs = torch.nonzero(label > 1).view(-1)
    masked_labels = label[object_idxs].int().cpu()
    coords = coords[object_idxs]
    batch_idxs = batch_idxs[object_idxs]
    batch_offsets = get_batch_offsets(batch_idxs, batch_size)
    
    coords = coords.float().contiguous()
    idx, start_len = pointgroup_ops.ballquery_batch_p(coords, batch_idxs, batch_offsets, cluster_radius, cluster_meanActive)

    proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(masked_labels, idx.cpu(), start_len.cpu(), cluster_npoint_thre)
    proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
    
    # print("proposals_idx:", proposals_idx)
    # c_idxs = proposals_idx[:, 1].cuda()
    # clusters_coords = coords[c_idxs.long()]
    
    # # assert False
    
    # cluster_label = proposals_idx[:, 0]
    
    return proposals_idx, proposals_offset

def GetClusterCenter(clusters_idx, clusters_offset, stu_feats, teach_feats):
    c_idxs = clusters_idx[:, 1].cuda()
    teach_feats = teach_feats[c_idxs.long()]
    stu_feats = stu_feats[c_idxs.long()]
    
    clusters_stu_feats = pointgroup_ops.sec_mean(stu_feats, clusters_offset.cuda())  # (nCluster, C), float
    clusters_teacher_feats = pointgroup_ops.sec_mean(teach_feats, clusters_offset.cuda())  # (nCluster, C), float
    
    return clusters_stu_feats, clusters_teacher_feats
