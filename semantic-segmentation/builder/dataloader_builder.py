import torch
import random
import numpy as np
from util.s3dis import S3DIS
from util.scannet_v2 import Scannetv2, Scannetv2_ME
from functools import partial
from util.data_util import collate_fn, collate_fn_limit, cfl_collate_fn_factory


def worker_init_fn(worker_id, manual_seed):
    random.seed(manual_seed + worker_id)
    np.random.seed(manual_seed + worker_id)
    torch.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed_all(manual_seed + worker_id)


def build_dataloader(args, prevoxel_transform_train, train_transform):

    manual_seed = args.manual_seed
    if args.data_name == 's3dis':
        train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    elif args.data_name == 'scannetv2':
        train_split = args.get("train_split", "trainval")
        train_data = Scannetv2(split=train_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    elif args.data_name == 'scannetv2_me':
        train_split = args.get("train_split", "train")
        return_ori_coord = args.get("use_ori_coords", False)
        train_data = Scannetv2_ME(split=train_split, data_root=args.data_root, voxel_size=args.voxel_size, teach_voxel_size=args.get('teach_voxel_size', None), prevoxel_transforms=prevoxel_transform_train, transform=train_transform, ignore_label=args.ignore_label, return_ori_coords=return_ori_coord, shuffle_index=True, distill_mode=args.get("distill_mode", None), loop=args.loop)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    
    if args.data_name == 'scannetv2_me':
        train_collate_fn = cfl_collate_fn_factory(args.max_batch_points)
    else:
        train_collate_fn = partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger= None)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=train_collate_fn, worker_init_fn=partial(worker_init_fn, manual_seed=args.manual_seed))
     
    
    val_loader = None
    if args.evaluate:
        val_transform = None
        if args.data_name == 's3dis':
            val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
        elif args.data_name == 'scannetv2':
            val_data = Scannetv2(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
        elif args.data_name == 'scannetv2_me':
            use_high_val = args.get("use_high_val", False)
            high_voxel_size=args.get('teach_voxel_size', None)
            if use_high_val:
                if high_voxel_size:
                    val_data = Scannetv2_ME(split='val', data_root=args.data_root, voxel_size=args.voxel_size, transform=val_transform, teach_voxel_size=high_voxel_size, return_ori_coords=args.get("use_ori_coords", False))
                else:
                    raise ValueError("high_voxel_size is None.")
            else:
                val_data = Scannetv2_ME(split='val', data_root=args.data_root, voxel_size=args.voxel_size, transform=val_transform)
        else:
            raise ValueError("The dataset {} is not supported.".format(args.data_name))

        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
            
        if args.data_name == 'scannetv2_me':
            val_collate_fn = cfl_collate_fn_factory(limit_numpoints=None)
        else:
            val_collate_fn = collate_fn
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, \
                pin_memory=True, sampler=val_sampler, collate_fn=val_collate_fn)
    
    return train_loader, val_loader, train_sampler, val_sampler
