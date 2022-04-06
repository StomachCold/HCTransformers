import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer_pooling as vits
import vit
from vision_transformer_attn import DINOHead
import warnings
warnings.filterwarnings("ignore")
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('HCTransformers-Pooling', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--pretrained_weights', default='/home/heyj/dino/checkpoint_kl250/checkpoint0393.pth',
        type=str, help="Path to pretrained weights.")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=30, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")


    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/mini_imagenet/train', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="checkpoint_triplet", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=777, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser




def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============

    # freeze_path = '/home/heyj/dino/checkpoint_kl250/checkpoint0393.pth'
    # ============ feature exctraction ... ============

    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
        is_student=True
    )
    embed_dim = student.embed_dim
    utils.load_pretrained_weights(student, args.pretrained_weights, 'student', args.arch,
                                  args.patch_size)

    student = utils.MultiCropWrapper_pooling(student, None, 'student', True)


    teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
    # utils.load_pretrained_weights(teacher, freeze_path, 'teacher', args.arch,args.patch_size)
    teacher = utils.MultiCropWrapper_pooling(teacher, None, 'teacher', True)


    student, teacher = student.cuda(), teacher.cuda()

    # ============ pooling*2 net ... ============

    student_392 = vit.vit_small(
        num_patches=392,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
        is_student=True
    )
    teacher_392 = vit.vit_small(
        num_patches=392,
    )
    student_392 = utils.MultiCropWrapper_pooling(student_392, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ), 'student',True)
    teacher_392 = utils.MultiCropWrapper_pooling(
        teacher_392,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head), 'teacher',True
    )

    # ============ pooling*4 net ... ============
    student_196 = vit.vit_small(
        num_patches=196,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
        is_student=True
    )
    teacher_196 = vit.vit_small(
        num_patches=196,
    )
    student_196 = utils.MultiCropWrapper_pooling(student_196, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ), 'student')
    teacher_196 = utils.MultiCropWrapper_pooling(
        teacher_196,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head), 'teacher'
    )

    #kk = torch.load(freeze_path, map_location="cpu")['student']
    #load_model(kk, student_392)
    #load_model(kk, student_196)
    #del kk
    student_392,student_196,teacher_392,teacher_196 = student_392.cuda(),student_196.cuda(),teacher_392.cuda(),teacher_196.cuda()


    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    student_392 = nn.parallel.DistributedDataParallel(student_392, device_ids=[args.gpu])
    student_196 = nn.parallel.DistributedDataParallel(student_196, device_ids=[args.gpu])

    # teacher and student start with the same weights
    teacher.load_state_dict(student.module.state_dict())
    teacher_392.load_state_dict(student_392.module.state_dict())
    teacher_196.load_state_dict(student_196.module.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in student.parameters():
        p.requires_grad = False
    for p in teacher.parameters():
        p.requires_grad = False
    for p in teacher_392.parameters():
        p.requires_grad = False
    for p in teacher_196.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")



    # ============ preparing loss ... ============
    dino_loss_392 = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    dino_loss_196 = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    surrogate_loss_392 = SurrogateLoss(args.local_crops_number + 2, num_classes=64, is_kl=True)
    surrogate_loss_392_2 = SurrogateLoss(2, feat_dim=384, num_classes=64, is_kl=True).cuda()
    surrogate_loss_196 = SurrogateLoss(args.local_crops_number + 2, num_classes=64, is_kl=True)
    surrogate_loss_196_2 = SurrogateLoss(2, feat_dim=384, num_classes=64, is_kl=True).cuda()
    #kk = torch.load(freeze_path, map_location="cpu")['surrogate_loss1']
    #load_model(kk, surrogate_loss_392)
    #load_model(kk, surrogate_loss_196)
    #del kk
    # surrogate_loss2 = SurrogateLoss(2, feat_dim=384,num_classes=64).cuda()
    surrogate_loss_392 = surrogate_loss_392.cuda()
    surrogate_loss_196 = surrogate_loss_196.cuda()
    # ============ preparing optimizer ... ============
    params_groups_392 = utils.get_params_groups(student_392)
    params_groups_196 = utils.get_params_groups(student_196)
    params_groups_392[0]['params'] = params_groups_392[0]['params'] + \
                                list(surrogate_loss_392.parameters()) + \
                                list(surrogate_loss_392_2.parameters())
    params_groups_196[0]['params'] = params_groups_196[0]['params'] + \
                                list(surrogate_loss_196.parameters()) + \
                                list(surrogate_loss_196_2.parameters())
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups_392 + params_groups_196)  # to use with ViTs
    # elif args.optimizer == "sgd":
    #     optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    # elif args.optimizer == "lars":
    #     optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        0.0002,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))


    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student_392=student_392,
        teacher_392=teacher_392,
        student_196=student_196,
        teacher_196=teacher_196,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss_392=dino_loss_392,
        dino_loss_196=dino_loss_196,
        surrogate_loss_392=surrogate_loss_392,
        surrogate_loss_392_2=surrogate_loss_392_2,
        surrogate_loss_196=surrogate_loss_196,
        surrogate_loss_196_2=surrogate_loss_196_2,

    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):

        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher,student_392,teacher_392,student_196,teacher_196, dino_loss_392,dino_loss_196,surrogate_loss_392,surrogate_loss_392_2,surrogate_loss_196,surrogate_loss_196_2,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student_392': student_392.state_dict(),
            'teacher_392': teacher_392.state_dict(),
            'student_196': student_196.state_dict(),
            'teacher_196': teacher_196.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss_392': dino_loss_392.state_dict(),
            'dino_loss_196': dino_loss_196.state_dict(),
            'surrogate_loss_392':surrogate_loss_392.state_dict(),
            'surrogate_loss_392_2':surrogate_loss_392_2.state_dict(),
            'surrogate_loss_196':surrogate_loss_196.state_dict(),
            'surrogate_loss_196_2':surrogate_loss_196_2.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            if epoch >= 270:
                args.saveckp_freq = 1
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.save_tb(log_stats)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




def train_one_epoch(student, teacher,student_392,teacher_392,student_196,teacher_196, dino_loss_392,dino_loss_196,surrogate_loss_392,surrogate_loss_392_2,surrogate_loss_196,surrogate_loss_196_2, data_loader,
                    optimizer, lr_schedule,  wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    # if epoch % 50 == 0 and epoch != 0:
    #     loss_schedule *= 0.3
    for it, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_tokens, teacher_labels = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_tokens, student_labels = student(images)
            # ==========================pooling * 2 ===========================
            #spectral cluster
            teacher_output, teacher_tokens, _ = teacher_392(teacher_tokens, labels=teacher_labels)
            student_output, student_tokens, student_pth = student_392(student_tokens, labels=student_labels)
            dinoloss392 = dino_loss_392(student_output,teacher_output,epoch)
            surrogateloss392_1 = surrogate_loss_392(student_output, labels)
            # surrogateloss392_2 = surrogate_loss_392_2(student_pth, labels) * 0.1
            # ==========================pooling * 4 ===========================
            teacher_output, _ = teacher_196(teacher_tokens)
            student_output, student_pth = student_196(student_tokens)
            dinoloss196 = dino_loss_196(student_output, teacher_output, epoch)
            surrogateloss196_1 = surrogate_loss_196(student_output, labels)
            # surrogateloss196_2 = surrogate_loss_196_2(student_pth, labels) * 0.1
            loss = dinoloss392 + surrogateloss392_1 + dinoloss196 + surrogateloss196_1
            # loss = dinoloss392 + surrogateloss392_1 + surrogateloss392_2 + dinoloss196 + surrogateloss196_1 + surrogateloss196_2
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                #nn.utils.clip_grad_value_(student.parameters(), 0.6)
                param_norms = utils.clip_gradients(student_392, args.clip_grad)
                param_norms = utils.clip_gradients(student_196, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student_392,
                                              args.freeze_last_layer)
            utils.cancel_gradients_last_layer(epoch, student_196,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            # torch.autograd.set_detect_anomaly(True)
            # with torch.autograd.detect_anomaly():
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student_392, args.clip_grad)
                param_norms = utils.clip_gradients(student_196, args.clip_grad)

            utils.cancel_gradients_last_layer(epoch, student_392,
                                              args.freeze_last_layer)
            utils.cancel_gradients_last_layer(epoch, student_196,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student_392.module.parameters(), teacher_392.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(student_196.module.parameters(), teacher_196.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(dinoloss_392=dinoloss392.item())
        metric_logger.update(clsloss_392=surrogateloss392_1.item())
        # metric_logger.update(clsloss_392_2=surrogateloss392_2.item())
        metric_logger.update(dinoloss_196=dinoloss196.item())
        metric_logger.update(clsloss_196=surrogateloss196_1.item())
        # metric_logger.update(clsloss_196_2=surrogateloss196_2.item())
        # if epoch <= 250:
        #     metric_logger.update(patchloss=surrogateloss2.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        # metric_logger.update(loss_rate=loss_schedule[it])

        # Save tb record every
        if (it+1) % 25 == 0 or (it+1) % len(data_loader) == 0:
            utils.save_tb_iter(
                dict(
                    loss=loss.item(),dinoloss_392=dinoloss392.item(),dinoloss_196=dinoloss196.item(),
                    clsloss_392=surrogateloss392_1.item(),clsloss_196=surrogateloss196_1.item()),
                it,
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # eval_epoch.save_features(teacher, epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 surrogate_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.surrogate_momentum = surrogate_momentum
        self.ncrops = ncrops
        self.register_buffer("surrogate", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher surrogateing and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.surrogate) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_surrogate(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_surrogate(self, teacher_output):
        """
        Update surrogate used for teacher output.
        """
        batch_surrogate = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_surrogate)
        batch_surrogate = batch_surrogate / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.surrogate = self.surrogate * self.surrogate_momentum + batch_surrogate * (1 - self.surrogate_momentum)


class SurrogateLoss(nn.Module):
    """Surrogate loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, ncrops, num_classes=64, feat_dim=8192, use_gpu=True,is_kl=False):
        super(SurrogateLoss, self).__init__()
        self.num_classes = num_classes
        self.is_kl=is_kl
        self.ncrops = ncrops
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.surrogates = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.surrogates = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """

        x = x.chunk(self.ncrops)
        assert x[0].size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        total_loss = 0
        if self.is_kl:
            for i in range(2):
                surrogate = self.surrogates[labels]
                kl = F.kl_div(F.log_softmax(x[i],dim=-1),F.softmax(surrogate,dim=-1),reduction='sum')
                loss = torch.clamp(kl, min=1e-5, max=1e+5).mean(dim=-1)
                total_loss += loss
            return total_loss/2
        else:
            for i in range(2):
                surrogate = self.surrogates[labels]
                dist = (x[i] - surrogate).pow(2).sum(dim=-1)
                loss = torch.clamp(dist, min=1e-5, max=1e+5).mean(dim=-1)
                total_loss += loss
            return total_loss/1000/2



class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        '''
        mini/tiered:   
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        cifar/FC100:  
            mean = [x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]]
            std = [x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]]      

        '''
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),

        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

def load_model(save_model,model):
    model_dict = model.state_dict()
    save_model = {k.replace('module.', ''): v for k, v in save_model.items() if 'pos_embed' not in k}
    # save_model = {k:v for k, v in save_model.items()}
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print('Successfully loaded', len(state_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)







