import torch
import torch.nn as nn

from torch.nn import functional as F
from torch_geometric.nn import voxel_grid

class PerspectiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, stu_p, teach_p):
        assert stu_p.shape[0] == teach_p.shape[0] # [num_classes1 + num_classes2 ..., C] 
        
        # print("s min:", stu_p.min(1)[0], "s max:", stu_p.max(1)[0], "s mean:", stu_p.mean(1), "t min:", teach_p.min(1)[0], "t max:", teach_p.max(1)[0], "t mean:", teach_p.mean(1))

        cos_dist = torch.cosine_similarity(stu_p, teach_p, dim=1)

        return 1.0 - torch.mean(cos_dist)

class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, stu_feat, teach_feat):
        assert stu_feat.shape[0] == teach_feat.shape[0] # [num_classes1 + num_classes2 ..., C] 
        
        # print("s min:", stu_p.min(1)[0], "s max:", stu_p.max(1)[0], "s mean:", stu_p.mean(1), "t min:", teach_p.min(1)[0], "t max:", teach_p.max(1)[0], "t mean:", teach_p.mean(1))

        cos_dist = torch.cosine_similarity(stu_feat, teach_feat, dim=1)

        return 1.0 - torch.mean(cos_dist)
        

def get_observation(feats, target, perspectives, ignore_idx=-100):
    feats = feats[target!=ignore_idx]
    # feats [N, C], persepectives [num_class, C]
    cos_dist = torch.cosine_similarity(feats.unsqueeze(1), perspectives.unsqueeze(0), dim=-1)
    # cos_dist [N, num_class]
    assert cos_dist.shape[-1] == perspectives.shape[0]
    return cos_dist

def grid_sample(pos, batch, size, start, return_p2v=True):
    # pos: float [N, 3]
    # batch: long [N]
    # size: float [3, ]
    # start: float [3, ] / None
    cluster = voxel_grid(pos, batch, size, start=start) #[N, ]

    if return_p2v == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)

    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)
    return cluster, p2v_map, counts

def get_cosinesimilarity(feats, target, perspectives, ignore_idx=-100, thre=None):
    # assert feats.shape[0] == target.shape[0]
    if target is not None:
        assert feats.shape[0] == target.shape[0]
        if thre is not None:
            # print("feats shape before:", feats.shape)
            object_idxs = torch.nonzero(target > thre).view(-1)
            feats = feats[object_idxs]
            # print("feats shape after:", feats.shape)
        else:
            feats = feats[target != ignore_idx]
    # feats [N, C], persepectives [num_class, C]
    # norm
    feats = F.normalize(feats, 2, 1).view(-1, feats.shape[1], 1) # (N, C, 1)
    perspectives = F.normalize(perspectives, 2, 1).view(-1, perspectives.shape[1], 1) # (num_class, C, 1)

    cos_dist = F.conv1d(input=feats, weight=perspectives, stride=1, padding=0).squeeze(-1)
    # print("cos_dist shape:", cos_dist.shape, "perspectives shape", perspectives.shape)
    assert cos_dist.shape[-1] == perspectives.shape[0]
    return cos_dist

def get_local_affinity_matrix(xyz, feats, batch, window_size, start=None):
    
    _, p2v_map, counts = grid_sample(xyz, batch, window_size, start=start)

    N, C = feats.shape
    n, k = p2v_map.shape

    mask = torch.arange(k).unsqueeze(0).cuda() < counts.unsqueeze(-1) #[n, k]
    mask_mat = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) #[n, k, k]
    # print("mask_mat shape:", mask_mat.shape, "window_size:", window_size)
    index_0 = p2v_map.unsqueeze(-1).expand(-1, -1, k)[mask_mat] #[M, ]
    index_1 = p2v_map.unsqueeze(1).expand(-1, k, -1)[mask_mat] #[M, ]
    
    affinity_matrix = torch.cosine_similarity(feats[index_0], feats[index_1])
    assert affinity_matrix.shape[0] == index_0.shape[0]
    
    return affinity_matrix

# def CriterionEntropySmooth(pred, soft, target, smoothness=0.5, eps=0, ignore_label=-100):
#     N, C = soft.shape
#     soft.detach()
#     onehot = target.view(-1, 1)
#     ignore_mask = (onehot==ignore_label).float()
#     onhot = onhot * (1 - ignore_mask)
#     onhot = torch.zeros(N, C).cuda().scatter_(1, onhot.long(), 1)
#     onhot = onehot.contiguous() # [N, C]
#     sm_soft = F.softmax(soft/1, 1)
#     smoothed_label = smoothness * sm_soft + (1 - smoothness) * onehot
#     if eps > 0:
#         smoothed_lebl = smoothed_label * (1 - eps) + (1 - smoothed_label) * eps / (smoothed_label.shape[1] - 1)
    
#     loss = torch.mul(-1 * F.log_softmax(pred, dim=1), smoothed_label) # [N, C]
    
    
    

    