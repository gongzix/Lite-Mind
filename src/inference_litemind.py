import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from tqdm import tqdm

from functools import partial

from dft_backbone import DFTBackbone

import random
from thop import profile


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
        help="gpu device number",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=450,
        help="patch size",
    )

    opt = parser.parse_args()
    patch_size = opt.patch_size
    device = 'cuda:'+opt.device
    subject=opt.subject

    
    X_te = np.load(f'./mrifeat/{subject}/{subject}_nsdgeneral_betas_ave_te.npy').astype('float32')
    Y_image = np.load(f'./imgfeat/{subject}_te.pth').astype("float32")

    X_te=torch.tensor(X_te).to(device)
    Y_image=torch.tensor(Y_image).to(device)


    print(f'Now infer model for... {subject}: ')
    print(f' fMRI:{X_te.shape}, Y_image:{Y_image.shape}')

    test_dataset = TensorDataset(X_te, Y_image)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        batch_size=1,
    )
    
    
    
    model = DFTBackbone(input_size=X_te.shape[-1], patch_size=patch_size, embed_dim=768, num_tokens=[512,256,128,257], depth=[2,10,2,4],
        mlp_ratio=[4, 4, 4, 4], drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    
    model.to(device)
    checkpoint=torch.load(f'./weight/litemind-{subject}.pth', map_location=device)

    model.load_state_dict(checkpoint)
    model.eval()
    
    preV=[]
    preV=torch.tensor(preV).to(device)

    import time
    with torch.no_grad():
        with tqdm(total=X_te.shape[0]) as t:
            for test_input, _ in test_loader:
                
                predictV = model(test_input)
                preV = torch.cat((preV, predictV), dim = 0)
  
                t.set_description('Infer')
                t.update(1)

        top1=0

        
        Y_image = Y_image / Y_image.norm(p=2, dim=-1, keepdim=True)
        preV = preV / preV.norm(p=2, dim=-1, keepdim=True)

        similarity = (100.0 * preV @ Y_image.T).softmax(dim=-1)

        top1 = 0
        for i in range(preV.shape[0]):
            _, indices = similarity[i].topk(1)
            if indices == i:
                top1+=1
        print(f'top1@982= {top1/982*100:.2f}%')

        avg_top = 0
        topmax = 0
        top_seed = 0
        print('Start loop with retrieval pool: 300......')
        for j in range(0,30):
            top1 = 0
            random.seed(j)
            similarity = (100.0 * preV @ Y_image.T).softmax(dim=-1)
            #similarity = (100.0 * Y_image @ preV.T).softmax(dim=-1)
            for i in range(preV.shape[0]):
                s = similarity[i]
                pool = list(range(982))
                pool.pop(i)
                selected_values = random.sample(pool, 682)
                s[selected_values] = 0.0
                _, indices = s.topk(1)
                if indices == i:
                    top1+=1
                    avg_top+=1

            print(f'loop {j},top1@300= {top1/982*100:.2f}%')
            if top1>topmax:
                topmax = top1
                top_seed = j
        print(f'max_seed={top_seed},top1max@300= {topmax/982*100:.2f}%')
        print(f'avg_top1@300={avg_top/(30*982)*100:.2f}%')
    
    
    

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    main()
