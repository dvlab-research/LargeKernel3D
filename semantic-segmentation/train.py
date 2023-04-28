import os
import time
import random
import numpy as np
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port

from util.logger import get_logger

from util.lr import MultiStepWithWarmup, PolyLR

from util.minkows_util import voxel_align_h, voxel_align_h_v2
from builder.transform_builder import build_transform_me
from builder.dataloader_builder import build_dataloader
import MinkowskiEngine as ME
import spconv.pytorch as spconv

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # get model
    if args.arch == "LargeKernel3D":
        from model import LargeKernel3D
        model = LargeKernel3D(3, 20)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    

    # set loss func 
    criterions = [nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()]

    # set optimizer
    opt_params = model.parameters()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(opt_params, lr=args.base_lr, momentum=args.momentum, dampening=args.get('dampening', 0.), weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(opt_params, lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in opt_params if "blocks" not in n and p.requires_grad]},
            {
                "params": [p for n, p in opt_params if "blocks" in n and p.requires_grad],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        k1, k2 = 0, 0
        for n, p in model.named_parameters():
            if "blocks" in n and p.requires_grad:
                print("n: ", n)
                k2 += 1
            else:
                k1 += 1
        print('k1: {}, k2: {}'.format(k1, k2))
        optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('# Student Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        print('args.batch_size', args.batch_size)
        print('ngpus_per_node', ngpus_per_node)
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        else:
            model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)        
    else:
        model = model.cuda()
        model.device = torch.device('cuda')

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            #best_iou = 40.0
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    ###################
    # build transform #
    ###################
    # train_transform = build_transform(args)
    
    prevoxel_transform_train, train_transform = build_transform_me(args)

    ###################
    # build dataloader #
    ###################
    train_loader, val_loader, train_sampler, val_sampler = build_dataloader(args, prevoxel_transform_train, train_transform)
    
    # set scheduler
    if args.scheduler == "MultiStepWithWarmup":
        assert args.scheduler_update == 'step'
        if main_process():
            logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
        iter_per_epoch = len(train_loader)
        milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
        scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
            warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    elif args.scheduler == 'MultiStep':
        assert args.scheduler_update == 'epoch'
        if main_process():
            logger.info("scheduler: MultiStep. scheduler_update: {}".format(args.scheduler_update))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)
    elif args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    scaler = None #torch.cuda.amp.GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))
            
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterions, optimizer, epoch, scaler, scheduler)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1
        
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):

            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterions[0])

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def train(train_loader, model, criterions, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, data_items in enumerate(train_loader):
        if args.get('teach_voxel_size', None):
            if args.get("use_ori_coords", False):
                coord, feat, target_s, ori_coords, offset, coord_t, feat_t, target_t, ori_coords_t, offset_t = data_items
            else:
                coord, feat, target_s, offset, coord_t, feat_t, target_t, offset_t = data_items
        else:
            coord, feat, target = data_items
        # offset 存储了每个batch的index指标
        data_time.update(time.time() - end)

        torch.cuda.empty_cache()
        
        coord[:, 1:] -= coord[:, 1:].min(dim=0)[0].unsqueeze(0)
        if args.get('teach_voxel_size', None):
            coord_t[:, 1:] += shift_size

        if args.data_name == "scannetv2_me":
            feat[:, :3] = feat[:, :3] / 255. - 0.5
            
        sinput = spconv.SparseConvTensor(feat.to(model.device), coord.to(model.device), coord[:, 1:].max(dim=0)[0], args.batch_size)

        if args.get('teach_voxel_size', None):
            target = target_t.cuda(non_blocking=True).long()
            offset = offset.cuda(non_blocking=True)
            offset_t = offset_t.cuda(non_blocking=True)
            if args.get("use_ori_coords", False):
                ori_coords = ori_coords.cuda(non_blocking=True)
                ori_coords_t = ori_coords_t.cuda(non_blocking=True)
            else:
                teach_input = ME.SparseTensor(feat_t, coord_t, device=model.device)
        else:
            target = target.cuda(non_blocking=True).long()

        soutput = model(sinput)[0].features

        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        if args.get('teach_voxel_size', None) and args.arch != "MinkUNet14TinyUpSample" and args.arch != "MinkUNet14TinyUpSampleNOXYZ":
            if args.get("use_ori_coords", False):
                soutput, _ = voxel_align_h_v2(ori_coords, ori_coords_t, None, soutput, offset, offset_t)
            else:
                soutput, _ = voxel_align_h(sinput, teach_input, None, soutput, offset, offset_t, args.voxel_size, args.teach_voxel_size, shift_size)

        loss = criterions[0](soutput, target)

        optimizer.zero_grad()
        loss.backward()

        if args.get("max_grad_norm", None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        output = soutput.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                print('lr', lr)
                lr = [round(float(x), 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          lr=lr,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc

def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()
    model.eval()
    
    end = time.time()
    for i, data_items in enumerate(val_loader):
        if args.get('use_high_val', False):
            if args.get("use_ori_coords", False):
                coord, feat, target_s, ori_coords, offset, coord_t, feat_t, target_t, ori_coords_t, offset_t = data_items
            else:
                coord, feat, target_s, offset, coord_t, feat_t, target_t, offset_t = data_items
        else:
            coord, feat, target = data_items
        data_time.update(time.time() - end)
        
        if args.data_name == "scannetv2_me":
            feat[:, :3] = feat[:, :3] / 255. - 0.5

        coord[:, 1:] -= coord[:, 1:].min(dim=0)[0].unsqueeze(0)

        sinput = spconv.SparseConvTensor(feat.to(model.device), coord.to(model.device), coord[:, 1:].max(dim=0)[0], args.batch_size)

        if args.get('use_high_val', False):
            target = target_t.cuda(non_blocking=True).long()
            offset = offset.cuda(non_blocking=True)
            offset_t = offset_t.cuda(non_blocking=True)
            if args.get("use_ori_coords", False):
                ori_coords = ori_coords.cuda(non_blocking=True)
                ori_coords_t = ori_coords_t.cuda(non_blocking=True)
            else:
                teach_input = ME.SparseTensor(feat_t, coord_t, device=model.device)
        else:
            target = target.cuda(non_blocking=True).long()

        
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls

        with torch.no_grad():
            soutput = model(sinput)[0].features
            loss = criterion(soutput, target)

        output = soutput.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
