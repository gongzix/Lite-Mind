# <center>Lite-Mind</center>

This is the official code for the paper "Lite-Mind:Towards Efficient and Robust Brain Representation Learning", which is an efficient model for fMRI decoding (<font color='blue' size=4>https://arxiv.org/abs/2312.03781</font>).[<font color='blue' size=4>**ACMMM 2024 Oral**</font>]


![model](assets/model.png)

## Data Preprocessing
1. Download `nsddata`, `nsddata_betas`, and `nsddata_stimuli` from NSD (http://naturalscenesdataset.org/) and place them under the nsd directory. 

2. Extraction of nsdgeneral roi from raw fMRI.
```python
python src/fmri2nsd.py --subject subj01
```

3. Extraction of features for the corresponding COCO images (Features can also be extracted at training time for data augmentation).
```python
python src/img2feat.py --subject subj01 --device 0 --nsddir ./nsd --savedir <your image feature save path>
```
## Train
Training can be done on a single RTX 4090.
```python
python src/train_litemind.py --device cuda:0 --patch-size 450 --batch-size 1000 --epochs 1500 --output_dir /home/students/gzx_4090_1/subj/subj07 --seed 42 --lr 1e-3 --featdir <your nsdgeneral path>  --weight-decay 0.1 --fmridir ./nsd_fsverage --subject subj01
```
## Inference on the Test Set
```python
python src/inference_litemind.py --device 0 --subject subj01 --model <your model path> 
```
![test_result](assets/test_result.png)

## LAION-5B Retrieval
![laion5b](assets/laion5b.png)