# HCTransformers

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attribute-surrogates-learning-and-spectral/few-shot-learning-on-mini-imagenet-5-way-1)](https://paperswithcode.com/sota/few-shot-learning-on-mini-imagenet-5-way-1?p=attribute-surrogates-learning-and-spectral)

PyTorch implementation for **"Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-shot Learning"**.  
[[`arxiv`](https://arxiv.org/abs/2203.09064v1)]

> Code will be continuously updated.

<div align="center">
  <img width="100%" alt="HCT Network Architecture" src=".github/network.png">
</div>

## Prerequisites
This codebase has been developed with Python version 3.8, [PyTorch](https://pytorch.org/) version 1.9.0, CUDA 11.1 and torchvision 0.10.0. It has been tested on Ubuntu 20.04. 

## Pretrained weights
Pretrained weights on 𝒎𝒊𝒏𝒊ImageNet, 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet, CIFAR-FS and FC100 are available now. Note that for `𝒕𝒊𝒆𝒓𝒆𝒅ImageNet` and `FC100` there are only checkpoints for the first stage (without cascaded training). Accuracy of 5-way 1-shot and 5-way 5-shot shown in the table is evaluated on the `test` split and for reference only. 

<table>
  <tr>
    <th>dataset</th>
    <th>1-shot</th>
    <th>5-shot</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>𝒎𝒊𝒏𝒊ImageNet</td>
    <td>71.16%</td>
    <td>84.60%</td>
    <td rowspan="4">
        <a href="https://cowtransfer.com/s/255a1df5901143">checkpoints_first</a>
    </td>
  </tr>
  <tr>
    <td>𝒕𝒊𝒆𝒓𝒆𝒅ImageNet</td>
    <td>79.67%</td>
    <td>91.72%</td>
  </tr>
  <tr>
    <td>FC100</td>
    <td>48.27%</td>
    <td>66.42%</td>
  </tr>
  <tr>
    <td>CIFAR-FS</td>
    <td>73.13%</td>
    <td>86.36%</td>
  </tr>
</table>

Pretrained weights for the **cascaded-trained models** on 𝒎𝒊𝒏𝒊ImageNet and CIFAR-FS are provided as follows. Note that *the path to pretrained weight in the first stage* must be specified when evaluating (see **Evaluation**).

<table>
  <tr>
    <th>dataset</th>
    <th>1-shot</th>
    <th>5-shot</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>𝒎𝒊𝒏𝒊ImageNet</td>
    <td>74.74%</td>
    <td>89.19%</td>
    <td rowspan="4">
        <a href="https://cowtransfer.com/s/6bd94675bca24c">checkpoints_pooling</a>
    </td>
  </tr>
  <tr>
    <td>CIFAR-FS</td>
    <td>78.89%</td>
    <td>90.50%</td>
  </tr>
</table>

## Datasets
~~Download links of datasets could be found in the **meta-transfer-learning** repo ([link](https://github.com/yaoyao-liu/meta-transfer-learning#datasets)).~~ We found that the image resolution in the off-the-shelf 𝒎𝒊𝒏𝒊ImageNet and 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet is 84 × 84, which is different from what we used (480 × 480). We apologize and regret for any inconvenience caused by our negligence.

### 𝒎𝒊𝒏𝒊ImageNet

> The 𝑚𝑖𝑛𝑖ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test. To generate this dataset from ImageNet, you may use the repository [𝑚𝑖𝑛𝑖ImageNet tools](https://github.com/y2l/mini-imagenet-tools).

Note that in our implemenation images are resized to 480 × 480 because the data augmentation we used require the image resolution to be greater than 224 to avoid distortions. Therefore, when generating 𝒎𝒊𝒏𝒊ImageNet, you should set ```--image_resize 0``` to keep the original size or ```--image_resize 480``` as what we did.

### 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet
> The [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet](https://arxiv.org/pdf/1803.00676.pdf) dataset is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy. To generate this dataset from ImageNet, you may use the repository 𝑡𝑖𝑒𝑟𝑒𝑑ImageNet dataset: [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet tools](https://github.com/y2l/tiered-imagenet-tools). 

Similar to 𝒎𝒊𝒏𝒊ImageNet, you should set ```--image_resize 0``` to keep the original size or ```--image_resize 480``` as what we did when generating 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet.


## Training
We provide the training code for 𝒎𝒊𝒏𝒊ImageNet, 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet and CIFAR-FS, extending the **DINO** repo ([link](https://github.com/facebookresearch/dino)). 


### 1 Pre-train the First Transformer
To pre-train the first Transformer with attribute surrogates learning on 𝒎𝒊𝒏𝒊ImageNet from scratch with multiple GPU, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_hct_first.py --arch vit_small --data_path /path/to/mini_imagenet/train --output_dir /path/to/saving_dir
```

### 2 Train the Hierarchically Cascaded Transformers
To train the Hierarchically Cascaded Transformers with sprectral token pooling on 𝒎𝒊𝒏𝒊ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_hct_pooling.py --arch vit_small --data_path /path/to/mini_imagenet/train --output_dir /path/to/saving_dir --pretrained_weights /path/to/pretrained_weights
```

## Evaluation
To evaluate the performance of the first Transformer on 𝒎𝒊𝒏𝒊ImageNet 5-way 1-shot task, run:
```
python eval_hct_first.py --arch vit_small --server mini --partition test --checkpoint_key student --ckp_path /path/to/checkpoint_mini/ --num_shots 1
```

To evaluate the performance of the Hierarchically Cascaded Transformers on 𝒎𝒊𝒏𝒊ImageNet 5-way 5-shot task, run:
```
python eval_hct_pooling.py --arch vit_small --server mini_pooling --partition val --checkpoint_key student --ckp_path /path/to/checkpoint_mini_pooling/  --pretrained_weights /path/to/pretrained_weights_of_first_satge --num_shots 5
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@misc{he2022hct,
  author    = {Yangji He, Weihan Liang, Dongyang Zhao, Hong-Yu Zhou, Weifeng Ge, Yizhou Yu, Wenqiang Zhang},
  title     = {Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-shot Learning},
  publisher = {arXiv},
  year      = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
