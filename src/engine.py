import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import ModelEma
import utils
import numpy as np



def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    #image_loss = contrastive_loss(similarity.t())
    #return (caption_loss + image_loss)/2
    return caption_loss

def ContrastiveLoss(fmri,image):

    logit_scale=torch.tensor([8]).to(fmri.device)

    fmri = fmri / fmri.norm(p=2, dim=-1, keepdim=True)
    image = image / image.norm(p=2, dim=-1, keepdim=True)

    logit_scale = logit_scale.exp()
    logits_per_fmri = torch.matmul(fmri, image.t()) * logit_scale
    #logits_per_fmri = torch.matmul(fmri, image.t())
    logits_per_image = logits_per_fmri.t()
    
    return clip_loss(logits_per_fmri)



def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, model_ema: Optional[ModelEma] = None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)     
        targets = targets.to(device, non_blocking=True)
        outputs = model(samples)
        loss = ContrastiveLoss(targets,outputs)
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    
        torch.cuda.synchronize(device)
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_prior(model: torch.nn.Module, diffusion_prior: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, model_ema: Optional[ModelEma] = None):

    diffusion_prior.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)     
        targets = targets.to(device, non_blocking=True)
        outputs = model(samples)
        loss, _ = diffusion_prior(text_embed=outputs.unsqueeze(1), image_embed=targets.unsqueeze(1))
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    
        torch.cuda.synchronize(device)
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    avg_top1 = 0
    avg_loss = 0
    with torch.no_grad():
        for test_input, target in metric_logger.log_every(data_loader, 10, header):
            test_input = test_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            #with torch.cuda.amp.autocast():
            output = model(test_input)
            
            top1=0
            output = output.flatten(1)
            output = output / output.norm(p=2, dim=-1, keepdim=True)
            target = target / target.norm(p=2, dim=-1, keepdim=True)
            
            similarity = (100.0 * output @ target.T).softmax(dim=-1)
            #similarity = (100.0 * target @output .T).softmax(dim=-1)
            
            for i in range(output.shape[0]):
                _, indices = similarity[i].topk(1)
                if indices == i:
                    top1+=1
            avg_top1 += top1

    return top1/3

@torch.no_grad()
def evaluate_prior(data_loader, model, diffusion_prior, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    diffusion_prior.eval()
    loss = 0
    with torch.no_grad():
        for test_input, target in metric_logger.log_every(data_loader, 10, header):
            test_input = test_input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            #with torch.cuda.amp.autocast():
            output = model(test_input)
            loss_prior, _ = diffusion_prior(text_embed=output.unsqueeze(1), image_embed=target.unsqueeze(1))
            loss += loss_prior

    return loss
