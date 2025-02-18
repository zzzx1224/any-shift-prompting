# Any-Shift Prompting for Generalization over Distributions

This repository provides the official PyTorch implementation of our CVPR 2024 paper:    

> Any-Shift Prompting for Generalization over Distributions
> 
> Zehao Xiao, Jiayi Shen, Mohammad Mahdi Derakhshani, Shengcai Liao, Cees G. M. Snoek

For more dtails, please check out our [<ins>**paper**</ins>](https://arxiv.org/abs/2402.10099). 

## Overview
This repository contains the implementation of Any-Shift Prompting for domain generalization tasks with a pre-trained CLIP.

## Prerequisites

### Hardware

This implementation is for the single-GPU configuration, evaluated on a single A6000. 

### Environments 
The code is tested on PyTorch 1.13.1. 

The code is based on [CoOp](https://github.com/KaiyangZhou/CoOp), with similar required packages to them, such as the [dassl](https://github.com/KaiyangZhou/Dassl.pytorch).

### Datasets 

We evaluate our method on benchmarks with different distribution shifts:

#### Covariate shifts
The model is trained on [ImageNet]((https://image-net.org/index.php) ) and evaluated on the following datasets:
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)
* [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

#### Label shifts
The model is trained on the base classes and evaluated on the novel classes for each following dataset:
* [ImageNet](https://image-net.org/index.php) 
* [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
* [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
* [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
* [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
* [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
* [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

#### Conditional shifts
We follow the conditional shifts setting in [BREEDS](https://arxiv.org/pdf/2008.04859), based on the [Living-17](https://github.com/MadryLab/BREEDS-Benchmarks) and [Entity-30](https://github.com/MadryLab/BREEDS-Benchmarks) datasets.
The dataset codes are provided by ```get_breeds``` function in [```dataset.py```](https://github.com/zzzx1224/any-shift-prompting/blob/main/Any-shift-Prompt-DG/dataset.py) file.

#### Concept shifts
We simulate the concept shifts by using ImageNet with hyperclasses. Codes are provided in [```imagenetcon.py```]() file.

## Training

You can run the following code for training our method:

The Base2New setting:
```bash
bash scripts/disp/base2new_train10.sh imagenet 1
```

The generalization setting:
```bash
bash scripts/disp/xd_train.sh imagenet 1
```

Change ```imagenet``` to other datasets for different evaluations.

Change ```1``` to other seeds.

Change the path ```DATA=YourOwnPath``` in the scripts file (```base2new_train10.sh``` and ```xd_train.sh```) to your own data path.

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
We thank the authors of [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) and [BREEDS](https://github.com/MadryLab/BREEDS-Benchmarks) for their open-source implementation and instructions on data preparation. 

