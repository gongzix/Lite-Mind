# <center>Lite-Mind</center>

This is the official code for the paper "Lite-Mind:Towards Efficient and Robust Brain Representation Learning", which is an efficient model for fMRI decoding (<font color='blue' size=4>https://arxiv.org/abs/2312.03781</font>).[<font color='blue' size=4>**ACM MM 2024 Oral**</font>]


![model](assets/model.png)

## Data Preprocessing
1. Download nsddata, nsddata_betas, and nsddata_stimuli from NSD and place them under the nsd directory. 

The NSD dataset website: http://naturalscenesdataset.org/

2. Extraction of nsdgeneral and spreading from raw fMRI
```python
python fmri2nsdgeneral.py --subject subj01
```

3. Extraction of features for the corresponding COCO images.
```python
python img2feat.py --subject subj01 --device 0 --nsddir ./nsd --savedir <your image feature save path>
```