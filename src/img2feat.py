import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io
import h5py  
import torch
from torchvision import transforms
import CLIP.clip as clip
from tqdm import tqdm
from PIL import Image


class RandomMask(object):
    ''' Apply random cutouts (masking) on an image'''
    def __init__(self, mask_ratio=0.5):
        ''' - mask_ratio: the ratio of (cutout area width or height) / (image width or height)'''
        self.cx = np.random.rand()
        self.cy = np.random.rand()
        self.m = mask_ratio / 2.0

    def __call__(self, sample):
        cx, cy, m = self.cx, self.cy, self.m
        _, x, y = sample.shape

        start_x = round((cx - m) * x)
        start_y = round((cy - m) * y)
        end_x = round((cx + m) * x)
        end_y = round((cy + m) * y)

        mask = torch.ones_like(sample)
        mask[:, max(0, start_x): min(x-1, end_x), max(0, start_y): min(y-1, end_y)] = 0

        return sample * mask

def get_img_trans(extra_aug=0.9, toPIL=True, img_size=256, color_jitter_p=0.4,
                  gray_scale_p=0.2, gaussian_blur_p=0.5, masking_p=1.0,
                  masking_ratio=0.0):
    '''
    - extra_aug: a value between 0-1. If 0, only apply resizing and to tensor.
                 If > 0, this p controls the probability that an augmentation
                 is actually implemented.
    - toPIL: bool. CLIP need PIL to process, if using other models, set it to F.
    - color_jitter_p: ADA 0.4, VICReg 0.8
    - gray_scale_p: VICReg 0.2
    - gaussian_blur_p: VICReg0.5, similar to ADA's filter.
                       (ADA has multiple, and has p = 1.0)
    - masking_ratio: ADA 0.5
    '''

    img_trans = []
    img_trans.append(transforms.ToTensor())
    img_trans.append(transforms.Resize((img_size, img_size)))

    run_extra = np.random.rand()
    if bool(extra_aug) and (run_extra < extra_aug):
        img_trans.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        img_trans.append(transforms.RandomHorizontalFlip(p=0.5))

        cj = np.random.rand()
        # print(f'color jitter {cj}, {cj < color_jitter_p}')
        if cj < color_jitter_p:
            img_trans.append(transforms.ColorJitter(0.4, 0.4, 0.2, 0.1))
        gs = np.random.rand()
        # print(f'grayscale {gs}, {gs < gray_scale_p}')
        if gs < gray_scale_p:
            img_trans.append(transforms.Grayscale(num_output_channels=3))
        gb = np.random.rand()
        # print(f'gaussian blur {gb}, {gb < gaussian_blur_p}')
        if gb < gaussian_blur_p:
            img_trans.append(transforms.GaussianBlur(kernel_size=23))
        # img_trans.append(transforms.RandomSolarize(128, p=0.1))

        img_trans.append(RandomMask(masking_ratio))

    if toPIL:
        img_trans.append(transforms.ToPILImage())
    img_trans = transforms.Compose(img_trans)
    return img_trans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    parser.add_argument(
        "--device",
        type=str,
        default='0',
        help="gpu",
    )

    parser.add_argument(
        "--nsddir",
        type=str,
        help="NSD file path you put",
    )

    parser.add_argument(
        "--savedir",
        type=str,
        help="Image CLIP features file path you wanna save",
    )

    opt = parser.parse_args()
    subject = opt.subject
    nsd=opt.nsddir
    device='cuda:'+opt.device
    savedir=opt.savedir
    
    nsda = NSDAccess(nsd)
    nsd_expdesign = scipy.io.loadmat(f'/{nsd}/nsddata/experiments/nsd/nsd_expdesign.mat')
    stim_file=f'{nsd}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'

    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    sharedix = nsd_expdesign['sharedix'] -1 

    behs = pd.DataFrame()
    for i in range(1,38):
        beh = nsda.read_behavior(subject=subject, 
                                session_index=i)
        behs = pd.concat((behs,beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_all = behs['73KID'] - 1
    stims_unique = behs['73KID'].unique() - 1

    visual_tr=torch.tensor([]).to(device)
    visual_te=torch.tensor([]).to(device)

    
    CLIP= clip.load("ViT-L/14", device)
    #CLIP= clip.load("ViT-B/32", device)
    with tqdm(total=len(stims_all)) as t:
        with h5py.File(stim_file, 'r') as f:
            for i in range (len(stims_all)):
                
                index=stims_all[i]
                if index in sharedix:
                    continue

                _image = f['imgBrick'][index]
                _image = get_img_trans(extra_aug=0)(_image)
                _image = CLIP[1](_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_image = CLIP[0].encode_image(_image).unsqueeze(0)
                    clip_image = clip_image[:,0:768]
                    #clip_image = clip_image.cpu()
                
                    
                    visual_tr=torch.cat((visual_tr,clip_image),dim=0)


                print(visual_tr.shape)
                
                t.set_description('stim %i' % i)
                t.update(1)

    
    with h5py.File(stim_file, 'r') as f:
        for idx,stim in enumerate(stims_unique):
                if stim in sharedix:
                    _image = f['imgBrick'][stim]
                    _image = get_img_trans(extra_aug=0)(_image)
                    _image = CLIP[1](_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        clip_image = CLIP[0].encode_image(_image).unsqueeze(0)
                        clip_image = clip_image[:,0:768]
                        #clip_image = clip_image.cpu()
                        visual_te=torch.cat((visual_te,clip_image),dim=0)
                    print(visual_te.shape)



    visual_tr=visual_tr.cpu().detach().numpy()
    visual_te=visual_te.cpu().detach().numpy()

    #np.save(f'{savedir}/image_clip_tr.npy',visual_tr)
    #np.save(f'{savedir}/image_clip_ave_te.npy',visual_te)
    visual_tr = torch.tensor(visual_tr)
    visual_te = torch.tensor(visual_te)
    torch.save(visual_tr, f'{savedir}/image_clip_tr.pth')
    torch.save(visual_te, f'{savedir}/image_clip_te.pth')
  


if __name__ == "__main__":
    main()
