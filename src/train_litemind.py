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

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma
from functools import partial
import torch.nn as nn

from engine import train_one_epoch, evaluate
import utils
from dft_backbone import DFTBackbone

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
    parser.add_argument('--drop_rate', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path_rate', type=float, default=0.15, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--clip_grad', type=float, default=1, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
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
    
    parser.add_argument('--model-ema', action='store_true')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')


    parser.add_argument('--output_dir', default='./weight',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    

    # NSD dataset and subject information
    parser.add_argument('--subject', default='subj01', help='subject number')
    parser.add_argument('--fmridir', default='../../testfmri', help='preprocessed fMRI file path')
    parser.add_argument('--featdir', default='../../nsdfeat/subjfeat', help='Image embeddings files from CLIP')
    parser.add_argument('--patch_size', default=450, type=int, help='patch_size')
    parser.add_argument('--percent', default=1.0, type=float, help='data percent')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False

    #load fMRI data and image features
    subject=args.subject
    patch_size=args.patch_size

    percent = args.percent


    X = np.load(f'./mrifeat/{subject}/{subject}_nsdgeneral_betas_tr.npy').astype('float32')
    X_te = np.load(f'./mrifeat/{subject}/{subject}_nsdgeneral_betas_ave_te.npy').astype('float32')
    X = torch.tensor(X)
    X_te=torch.tensor(X_te)

    Y = torch.load(f'./imgfeat/{subject}_tr.pth')
    Y_te = torch.load(f'./imgfeat/{subject}_te.pth')
    

    if percent<1.0:
        total_size = X.shape[0]
        indices = torch.randperm(total_size)
        numbers = int(total_size * percent)
        indices = indices[0:numbers]
        X = X[indices]
        Y = Y[indices]
    
    train_dataset = fMRIimageDataset(X, Y)
    test_dataset = fMRIimageDataset(X_te, Y_te)


    print(f'Now Train model for... {subject}')
    print(f' X: {X.shape}, Y :{Y.shape}')


    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)


    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        #batch_size=int(1.5 * args.batch_size),
        batch_size=300,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


        
    model = DFTBackbone(input_size=X.shape[-1], patch_size=patch_size, embed_dim=768, num_tokens=[512,256,128,257], depth=[2,10,2,4],
        mlp_ratio=[4, 4, 4, 4], drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6))

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
    output_dir = Path(args.output_dir)


    start_time = time.time()
    max_test=-9999

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn=None,
            set_training_mode=args.finetune == '' 
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device)



        if test_stats>=max_test:
            max_test=test_stats
            torch.save(model.state_dict(),os.path.join(output_dir,f'litemind-{subject}.pth'))
            max_epoch=epoch
        
            

        lr=train_stats['lr']
        loss=train_stats['loss']
        log_stats = f'train_lr:{lr},train_loss:{loss:.5f},epoch:{epoch},n_parameters:{n_parameters},Top1:{test_stats/982*100:.2f}%, epochmax:{max_epoch}'

        if args.output_dir and utils.is_main_process():
            with (output_dir / f"{subject}_log.txt").open("a") as f:
                f.write(log_stats+ "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Lite-Mind training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
