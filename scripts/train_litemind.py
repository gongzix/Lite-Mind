import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import random

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from functools import partial
import torch.nn as nn

from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils
from gfnet1d import GFNet1d, GFNet1dPyramid, BrainMLP, GFNet1dMLP
from FreDenoise import FreBrain

import warnings
warnings.filterwarnings("ignore", message="Argument interpolation should be")

class fMRIimageDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item, :],self.y[item, :]

    def __len__(self):
        return self.x.shape[0]



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str,
                        help='Name of model to train')
    parser.add_argument('--trial_size', default=24980, type=int, help='fMRI single trial input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    

    # NSD dataset and subject information
    parser.add_argument('--subject', default='subj01', help='subject number')
    parser.add_argument('--roi', default='early',nargs="*", help='Brain ROI')
    parser.add_argument('--fmridir', default='../../testfmri', help='preprocessed fMRI file path')
    parser.add_argument('--featdir', default='../../nsdfeat/subjfeat', help='Image embeddings files from CLIP')
    parser.add_argument('--kernel', default=10, type=int, help='kernel size')
    parser.add_argument('--stride', default=7, type=int, help='stride length')
    parser.add_argument('--percent', default=1.0, type=float, help='data percent')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False

    #load fMRI data and image features
    subject=args.subject
    roi=args.roi
    kernel_size=args.kernel
    stride=args.stride
    fmridir =args.fmridir
    featdir =args.featdir
    percent = args.percent


    X = np.load(f'/home/students/gzx_4090_1/StableDiffusionReconstruction-main/nsd_fsaverage/{subject}/{subject}_nsdgeneral_betas_tr.npy').astype('float32')
    X_te = np.load(f'/home/students/gzx_4090_1/StableDiffusionReconstruction-main/nsd_fsaverage/{subject}/{subject}_nsdgeneral_betas_ave_te.npy').astype('float32')
    X = torch.tensor(X)
    X_te=torch.tensor(X_te)
 
    if args.arch == 'gfnet-image':
        Y = torch.load(f'/home/students/gzx_4090_1/StableDiffusionReconstruction-main/nsd_fsaverage/{subject}_tr.pth')
        Y_te = torch.load(f'/home/students/gzx_4090_1/StableDiffusionReconstruction-main/nsd_fsaverage/{subject}_te.pth')
        #Y = np.load(f'/home/students/gzx_4090_1/GOD/{subject}/image_tr.npy')
        #Y_te = np.load(f'/home/students/gzx_4090_1/GOD/{subject}/image_te.npy')
    elif args.arch == 'gfnet-text':
        Y = np.load(f'{featdir}/text_clip_tr.npy').astype("float32")
        Y_te = np.load(f'{featdir}/text_clip_ave_te.npy').astype("float32")
    else:
        raise NotImplementedError

    

    if percent<1.0:
        total_size = X.shape[0]
        indices = torch.randperm(total_size)
        numbers = int(total_size * percent)
        indices = indices[0:numbers]
        X = X[indices]
        Y = Y[indices]
    
    train_dataset = fMRIimageDataset(X, Y)
    test_dataset1 = fMRIimageDataset(X_te, Y_te)


    print(f'Now Train model for... {subject}:  {roi}')
    print(f' X: {X.shape}, Y :{Y.shape}')

    #sampler_train = torch.utils.data.RandomSampler(train_dataset)
    #sampler_val = torch.utils.data.RandomSampler(test_dataset)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        sampler_val1 = torch.utils.data.SequentialSampler(test_dataset1)
        '''
        sampler_val2 = torch.utils.data.SequentialSampler(test_dataset2)
        sampler_val5 = torch.utils.data.SequentialSampler(test_dataset5)
        sampler_val7 = torch.utils.data.SequentialSampler(test_dataset7)
        '''
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val1 = torch.utils.data.DataLoader(
        test_dataset1, sampler=sampler_val1,
        #batch_size=int(1.5 * args.batch_size),
        batch_size=300,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    '''
    data_loader_val2 = torch.utils.data.DataLoader(
        test_dataset2, sampler=sampler_val2,
        #batch_size=int(1.5 * args.batch_size),
        batch_size=982,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val5 = torch.utils.data.DataLoader(
        test_dataset5, sampler=sampler_val5,
        #batch_size=int(1.5 * args.batch_size),
        batch_size=982,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val7 = torch.utils.data.DataLoader(
        test_dataset7, sampler=sampler_val7,
        #batch_size=int(1.5 * args.batch_size),
        batch_size=982,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    '''

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print('standard mix up')
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=512)
    else:
        print('mix up is not used')

    print(f"Creating model: {args.arch}")

    if args.arch == 'gfnet-image':
        
        '''
        model = GFNet1d(input_size=X.shape[1], kernel_size=kernel_size, stride=stride, in_chans=1, features=768, embed_dim=257, depth=12,
             mlp_ratio=4., representation_size=None, uniform_drop=False,
             drop_rate=0.1, drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
             dropcls=0
        )
        
        model = GFNet1dPyramid(input_size=X.shape[1], kernel_size=kernel_size, stride=stride, in_chans=1, features=512, embed_dim=[257], depth=[18],
            mlp_ratio=[4, 4, 4, 4], representation_size=None, uniform_drop=False,
            drop_rate=0.1, drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            dropcls=0
            )
        '''
        
        
        
        
        model = GFNet1dPyramid(input_size=X.shape[-1], kernel_size=kernel_size, stride=stride, in_chans=1, features=768, embed_dim=[512,256,128,257], depth=[2,10,2,4],
            mlp_ratio=[4, 4, 4, 4], representation_size=None, uniform_drop=False,
            drop_rate=0.1, drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            dropcls=0
            )
        
        
        
        '''
        model = GFNet1dMLP(input_size=X.shape[-1], kernel_size=kernel_size, stride=stride, in_chans=1, features=768, embed_dim=[257], depth=[1],
            mlp_ratio=[4, 4, 4, 4], representation_size=None, uniform_drop=False,
            drop_rate=0.1, drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            dropcls=0
        )
        '''
        #model = FreBrain(X.shape[-1], 257)
        
        
        #model = BrainMLP(out_dim =257*768,in_dim =X.shape[-1],clip_size =768,h=4096)

        
        
        
        
    elif args.arch == 'gfnet-text':
        '''
        model = GFNet1d(input_size=X.shape[1], kernel_size=kernel_size, stride=stride, in_chans=1, features=512, embed_dim=256, depth=12,
             mlp_ratio=4., representation_size=None, uniform_drop=False,
             drop_rate=0.1, drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
             dropcls=0
        )
        
        model = GFNet1dPyramid(input_size=X.shape[-1], kernel_size=kernel_size, stride=stride, in_chans=1, features=768, embed_dim=[1024,512,512,50], depth=[2,10,4,2],
            mlp_ratio=[4, 4, 4, 4], representation_size=None, uniform_drop=False,
            drop_rate=0.1, drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            dropcls=0
            )
        '''
        

    else:
        raise NotImplementedError

    

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    #linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    linear_scaled_lr = args.lr * 600 * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'


    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('lr scheduler will not be updated')
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_test1=-9999
    max_test2=0
    max_test5=0
    max_test7=0

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn=None,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)
        '''
        if epoch <=6:
            lr_scheduler.step(epoch)
        if epoch>0  and epoch % 200 == 0 :
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*0.5
        '''
        test_stats1, loss_mlp = evaluate(data_loader_val1, model, device)
        '''
        test_stats2 = evaluate(data_loader_val2, model, device)
        test_stats5 = evaluate(data_loader_val5, model, device)
        test_stats7 = evaluate(data_loader_val7, model, device)
        '''


        if test_stats1>=max_test1:
            max_test1=test_stats1
            torch.save(model.state_dict(),os.path.join(output_dir,f'GFNet-besttest-subj01.pth'))
            max_epoch1=epoch
        
        '''
        if test_stats2>=max_test2:
            max_test2=test_stats2
            torch.save(model.state_dict(),os.path.join(output_dir,f'GFNet-besttest-subj02.pth'))
            max_epoch2=epoch

        if test_stats5>=max_test5:
            max_test5=test_stats5
            torch.save(model.state_dict(),os.path.join(output_dir,f'GFNet-besttest-subj05.pth'))
            max_epoch5=epoch

        if test_stats7>=max_test7:
            max_test7=test_stats7
            torch.save(model.state_dict(),os.path.join(output_dir,f'GFNet-besttest-subj07.pth'))
            max_epoch7=epoch
        '''
        '''
        if epoch<31:
            torch.save(model.state_dict(),os.path.join(output_dir,f'GFNet-{epoch}.pth'))
        '''
            

        lr=train_stats['lr']
        loss=train_stats['loss']
        log_stats = f'train_lr:{lr},train_loss:{loss:.5f},epoch:{epoch},n_parameters:{n_parameters},Top1:{test_stats1/982*100:.2f}%, mse:{loss_mlp:.4f}, epochmax:{max_epoch1}'

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(log_stats+ "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GFNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
