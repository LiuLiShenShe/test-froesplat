import os
import sys
import math
import time
import json
import torch
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from typing import Sized
from pathlib import Path
from shutil import copyfile
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from amb3r.model import AMB3R

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

import croco.utils.misc as misc

from amb3r.datasets import *
from amb3r.loss import MultitaskLoss
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from vggt.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from moge.moge.train.losses import scale_invariant_alignment


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_resolution_by_gpu():
    gpu_name = torch.cuda.get_device_name(0).lower()
    if '4090' in gpu_name:
        num_frames = list(range(2, 5))
        num_frames_test = 10
        res_str = "[(518, 392), (518, 336), (518, 294), (518, 266), (518, 210), (518, 154)]"
        test_res_str = "(518, 392)"
        trainset = f"2000 @ Scannet(split='train', ROOT='./data/scannet/', resolution={res_str}, num_seq=1, num_frames={num_frames})"
        testset = f"Scannet(split='test', ROOT='./data/scannet/', resolution={test_res_str}, num_seq=1, num_frames={num_frames_test})"
        batch_test = 1
        return res_str, num_frames, trainset, testset, batch_test
    else:
        num_frames = list(range(5, 16))
        num_frames_test = 10
        res_str = "[(518, 392), (518, 336), (518, 294), (518, 266), (518, 210), (518, 154)]"
        test_res_str = "(518, 392)"
        batch_test = 1
        trainset = f"2000 @ Scannet(split='train', ROOT='./data/scannet/', resolution={res_str}, num_seq=1, num_frames={num_frames})"
        testset = f"Scannet(split='test', ROOT='./data/scannet/', resolution={test_res_str}, num_seq=1, num_frames={num_frames_test})"
        return res_str, num_frames, trainset, testset, batch_test


def get_args_parser():
    resolution, num_frames, trainset, testset, batch_size_test = get_resolution_by_gpu()

    parser = argparse.ArgumentParser('AMB3R metric scale training', add_help=False)
    parser.add_argument('--model', default="AMB3R(metric_scale=True)",
                        type=str, help="string containing the model to build")
    parser.add_argument('--interp_v2', action='store_true', default=False,
                        help='Expected interp_v2 setting; must match the pretrained checkpoint')

    parser.add_argument('--pretrained', required=True, help='path of an AMB3R checkpoint produced by training.py')

    parser.add_argument('--train_dataset',
                        default=trainset,
                        required=False, type=str, help="training set")
    parser.add_argument('--test_dataset',
                        default=testset,
                        required=False, type=str, help="test set")

    parser.add_argument('--train_criterion', default="MultitaskLoss()")
    parser.add_argument('--test_criterion', default="MultitaskLoss()")

    # Exp
    parser.add_argument('--seed', default=0, type=int, help="Random seed")

    # Training
    parser.add_argument('--batch_size', default=4, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--batch_size_test', default=batch_size_test, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=6, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=40, type=int, help="Maximum number of epochs for the scheduler")

    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', choices=[False, "bf16", "fp16"], default="bf16",
                        help="Use Automatic Mixed Precision for pretraining")

    # others
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--num_workers_test', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=10, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    parser.add_argument('--output_dir', default='./outputs/exp_amb3r_metric', type=str,
                        help="path where to save the output")

    return parser


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    dtype = get_dtype(args)

    for _, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        _, views_all = batch

        images = views_all['images'].to(device)
        key_depth = views_all['depthmap'][0, 0].to(device)  # H, W
        valid_key_mask = key_depth > 0

        assert images.shape[0] == 1, "only support bs=1 for now"

        with torch.autocast("cuda", dtype=dtype):
            pred_all = model.forward({'images': images})

        pred_key_depth = pred_all[-1]['depth_metric'][0, 0, ..., 0]  # H, W

        abs_rel = torch.nan_to_num(torch.abs(pred_key_depth - key_depth) / key_depth, nan=0.0)
        abs_rel_m = abs_rel[valid_key_mask].mean() * 100

        ratio = torch.max(
            torch.nan_to_num(key_depth / pred_key_depth, nan=2.0, posinf=2.0, neginf=2.0),
            torch.nan_to_num(pred_key_depth / key_depth, nan=2.0, posinf=2.0, neginf=2.0),
        )
        inlier125 = (ratio < 1.25).float()[valid_key_mask].mean() * 100

        metric_logger.update(abs_rel=float(abs_rel_m))
        metric_logger.update(inlier125=float(inlier125))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results


def get_dtype(args):
    if args.amp:
        dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    else:
        dtype = torch.float32
    return dtype


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    dtype = get_dtype(args)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        views, views_all = batch

        for key in views_all.keys():
            views_all[key] = views_all[key].to(device)

        with torch.autocast("cuda", dtype=dtype):
            pred_all = model.forward(views_all)

        views_all['extrinsics'], _, views_all['pts3d'], views_all['depthmap'], _ = normalize_camera_extrinsics_and_points_batch(
            extrinsics=views_all['extrinsics'],
            cam_points=None,
            world_points=views_all['pts3d'],
            depths=views_all['depthmap'],
            scale_by_points=False,
            point_masks=views_all['valid_mask'],
            pred_points=None
        )

        loss = 0.
        for iter in range(len(pred_all)):
            Bs, T, H, W, _ = views_all['pts3d'].shape

            pred_scale = pred_all[iter]['median_metric_z_log']
            pred_depth = pred_all[-1]['depth']
            pred_metric_depth = pred_all[iter]['depth_metric']
            gt_depth = views_all['depthmap'][..., None]
            valid_mask = views_all['valid_mask']

            # Find GT depth value at predicted median depth location
            pred_reshaped = pred_depth.view(Bs * T, H * W)
            gt_reshaped = gt_depth.view(Bs * T, H * W)
            median_pred_values_flat, _ = torch.median(pred_reshaped, dim=1)
            abs_diff = torch.abs(pred_reshaped - median_pred_values_flat.unsqueeze(1))
            median_indices_flat = torch.argmin(abs_diff, dim=1)
            corresponding_gt_values_flat = torch.gather(gt_reshaped, 1, median_indices_flat.unsqueeze(1)).squeeze(1)

            corresponding_gt_values = corresponding_gt_values_flat.view(Bs, T, 1)
            mask_reshaped = valid_mask.view(Bs * T, H * W)
            validity_of_median_gt_flat = torch.gather(mask_reshaped, 1, median_indices_flat.unsqueeze(1)).squeeze(1)
            valid_median_mask = validity_of_median_gt_flat.view(Bs, T, 1)

            # Fall back to scale-invariant alignment estimate for invalid pixels
            _, scale_depth = scale_invariant_alignment(
                pred_all[iter]['depth'].repeat(1, 1, 1, 1, 3),
                views_all['depthmap'][..., None].repeat(1, 1, 1, 1, 3),
                views_all['valid_mask'],
                trunc=1.0, detach=False)

            corresponding_gt_values[~valid_median_mask] = (
                median_pred_values_flat.view(Bs, T, 1)[~valid_median_mask]
                * scale_depth.view(Bs, T, 1)[~valid_median_mask].detach()
            )

            gt_log_z = torch.log(corresponding_gt_values + 1e-8)

            per_element_loss = F.l1_loss(pred_scale, gt_log_z, reduction='none')
            loss_iter = per_element_loss.mean()
            loss += loss_iter

            loss_depth = F.l1_loss(pred_metric_depth[valid_mask], gt_depth[valid_mask], reduction='mean')

            epoch_1000x = int(epoch_f * 1000)
            metric_logger.update(**{f'loss_metric{iter}': float(loss_iter.item())})
            metric_logger.update(**{f'loss_depth{iter}': float(loss_depth.item())})
            if log_writer is None:
                continue

            log_writer.add_scalar(f'train_loss_metric{iter}', loss_iter.item(), epoch_1000x)

        loss /= len(pred_all)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                           update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=1.0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss, pred_all, batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is None:
                continue

            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print('Building train dataset {:s}'.format(args.train_dataset))
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size_test, args.num_workers_test, test=True)
                        for dataset in args.test_dataset.split('+')}

    assert os.path.isfile(args.pretrained), f"pretrained checkpoint not found: {args.pretrained}"

    # Read interp_v2 from checkpoint before building the model
    ckpt = torch.load(args.pretrained, map_location='cpu')
    ckpt_state_dict = ckpt['model']
    ckpt_interp_v2 = ckpt_state_dict.get('backend.interp_v2', torch.tensor(False)).item()
    if ckpt_interp_v2 != args.interp_v2:
        raise ValueError(
            f"--interp_v2={args.interp_v2} disagrees with pretrained checkpoint (interp_v2={ckpt_interp_v2})"
        )

    model_str = args.model
    if ckpt_interp_v2 and 'interp_v2' not in model_str:
        model_str = model_str.rstrip(')') + ', interp_v2=True)'
    print('Loading model: {:s}'.format(model_str))
    model = eval(model_str)

    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    test_criterion = eval(args.test_criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if not args.resume:
        print('Loading pretrained: ', args.pretrained)
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in ckpt_state_dict.items()
                               if k in model_state_dict and v.shape == model_state_dict[k].shape}
        loading_info = model.load_state_dict(filtered_state_dict, strict=False)
        for key in loading_info.missing_keys:
            print(f"Missing key: {key}")
    del ckpt

    dtype = get_dtype(args)
    model.front_end.model.aggregator.to(dtype)

    for name, param in model.named_parameters():
        param.requires_grad = 'metric_scale' in name

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    params_to_train = [param for name, param in model_without_ddp.named_parameters() if 'metric_scale' in name]
    param_names_to_train = [name for name, param in model_without_ddp.named_parameters() if 'metric_scale' in name]
    print("Parameters being trained:")
    for name in param_names_to_train:
        print(name)

    optimizer = torch.optim.AdamW(params_to_train, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    best_so_far, best_relative_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                                        optimizer=optimizer, loss_scaler=loss_scaler)

    if best_so_far is None:
        best_so_far = float('inf')
    if best_relative_so_far is None:
        best_relative_so_far = 0.0

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    file_path_all = ['./', 'amb3r']
    os.makedirs(os.path.join(args.output_dir, 'recording'), exist_ok=True)
    for file_path in file_path_all:
        cur_dir = os.path.join(args.output_dir, 'recording', file_path)
        os.makedirs(cur_dir, exist_ok=True)
        files = os.listdir(file_path)
        for f_name in files:
            if f_name[-3:] == '.py':
                copyfile(os.path.join(file_path, f_name), os.path.join(cur_dir, f_name))

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()
        test_stats = {}

        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, 'last', best_so_far)

        # Test on multiple datasets
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            test_loss = 0.0
            test_relative_loss = 0.0
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats
                test_loss += stats['abs_rel_avg']
                test_relative_loss += stats['inlier125_avg']

            test_loss /= len(data_loader_test)
            test_relative_loss /= len(data_loader_test)

            if test_loss < best_so_far:
                best_so_far = test_loss
                new_best = True
                print(f"New best loss: {best_so_far:.4f} at epoch {epoch}")

            if test_relative_loss > best_relative_so_far:
                best_relative_so_far = test_relative_loss
                save_model(epoch - 1, f'best_relative_{best_relative_so_far}', best_relative_so_far)

        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, 'best', best_so_far)

        if epoch >= args.epochs:
            break

        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
