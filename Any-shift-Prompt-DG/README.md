# Any-Shift Prompting for Generalization over Distributions

This repository provides the official PyTorch implementation of our CVPR 2024 paper:    

> Any-Shift Prompting for Generalization over Distributions
> 
> Zehao Xiao, Jiayi Shen, Mohammad Mahdi Derakhshani, Shengcai Liao, Cees G. M. Snoek

For more details, please check out our [<ins>**paper**</ins>](https://arxiv.org/abs/2402.10099). 

## Overview
This repository contains the implementation of Any-Shift Prompting for domain generalization tasks with a pre-trained CLIP.

## Prerequisites

### Hardware

This implementation is for the single GPU configuration, evaluated on a single A6000. 

### Environments 
The code is tested on PyTorch 1.13.1. 

The code is based on [CoOp](https://github.com/KaiyangZhou/CoOp), with similar required packages to them, such as the [dassl](https://github.com/KaiyangZhou/Dassl.pytorch).

### Datasets 

For the common domain generalization setting, we consider 4 datasets:

* [PACS](https://domaingeneralization.github.io/#page-top)
* [VLCS](https://github.com/belaalb/G2DM#download-vlcs)
* [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
* [DomainNet](https://ai.bu.edu/M3SDA/)

Change ```data_path``` in ```ebm_dataset.py``` to your own data path.

## Training

You can run the code for training our method by running the following command:

```bash
python openmain.py --dataset PACS --test_domain sketch --cfgfile dg_config.yaml --log_dir pacs_sket --lr 5e-4 --vpro 1 --batch_size 10  --max_ite 3000  --opendg 1 --seed 42
```

Change the `opendg` to `0` for close set domain generalization.

## Citation
If you find our code useful or our work relevant, please consider citing: 
```
@inproceedings{xiao2024any,
  title={Any-Shift Prompting for Generalization over Distributions},
  author={Xiao, Zehao and Shen, Jiayi and Derakhshani, Mohammad Mahdi and Liao, Shengcai and Snoek, Cees GM},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13849--13860},
  year={2024}
}
```

## Acknowledgements
We thank the authors of [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) for their open-source implementation and instructions on data preparation. 
