<p align="center">
  <h2 align="center"> <img src="https://github.com/ranrhuang/ranrhuang.github.io/raw/master/spfsplat/static/image/icon.png" width="20" style="position: relative; top: 1px;"> No Pose at All  <br> Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views</h2>
 <p align="center">
    <a href="https://ranrhuang.github.io/">Ranran Huang</a>
    ·
    <a href="https://www.imperial.ac.uk/people/k.mikolajczyk">Krystian Mikolajczyk</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2508.01171">Paper</a> | <a href="https://ranrhuang.github.io/spfsplat/">Project Page</a>  </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/ranrhuang/ranrhuang.github.io/raw/master/spfsplat/static/image/framework.svg" alt="Teaser" width="90%">
  </a>
</p>


<p align="center">
<strong>SPFSplat</strong> simultaneous predicts 3D Gaussians and camera poses in a canonical space  <br> from unposed sparse images, requiring no ground-truth poses during training or inference.
</p>
<br>

<br>



<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a>
    </li>
    <li>
      <a href="#camera-conventions">Camera Conventions</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#running-the-code">Running the Code</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
</ol>
</details>

## Installation

1. Clone SPFSplat.
```bash
git clone https://github.com/ranrhuang/SPFSplat
cd SPFSplat
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n spfsplat python=3.11
conda activate spfpslat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
cd src/model/encoder/backbone/croco/curope/
python setup.py build_ext --inplace
cd ../../../../../..
```

## Pre-trained Checkpoints
Our models are hosted on [Hugging Face](https://huggingface.co/RanranHuang/SPFSplat) 🤗

|                                                    Model name                                                    | Training resolutions | Training data | Training settings |
|:----------------------------------------------------------------------------------------------------------------:|:--------------------:|:-------------:|:-------------:|
|                 [re10k.ckpt]( https://huggingface.co/RanranHuang/SPFSplat/resolve/main/re10k.ckpt)                  |        256x256       |     re10k     | RE10K, 2 views |
|                  [acid.ckpt]( https://huggingface.co/RanranHuang/SPFSplat/resolve/main/acid.ckpt )                  |        256x256       |     acid      | ACID, 2 views |
|         [re10k_dl3dv.ckpt]( https://huggingface.co/RanranHuang/SPFSplat/resolve/main/re10k_dl3dv.ckpt )         |        256x256       | re10k, dl3dv  | RE10K + DL3DV, 2 views |
|         [re10k_10view.ckpt]( https://huggingface.co/RanranHuang/SPFSplat/resolve/main/re10k_10view.ckpt)         |        256x256       | re10k | RE10K, 10 views |
|         [re10k_nointrin.ckpt]( https://huggingface.co/RanranHuang/SPFSplat/resolve/main/re10k_nointrin.ckpt)         |        256x256       | re10k | RE10K, w/o intrin embed., 2 views |

We assume the downloaded weights are located in the `pretrained_weights` directory.



## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## Running the Code
### Training
1. Download the [MASt3R](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) pretrained model and put it in the `./pretrained_weights` directory.

2. Train with:

```bash
# 2 view
python -m src.main +experiment=spfsplat/re10k wandb.mode=online wandb.name=re10k

# For multi-view training, we suggest fine-tuning from the released model. Here we use 3 view as an example. Remember to adjust the batch size according to your available GPU memory. 
python -m src.main +experiment=spfsplat/re10k_3view wandb.mode=online wandb.name=re10k_3view checkpointing.load=./pretrained_weights/re10k.ckpt checkpointing.resume=false 

# To inference without known intrinsics, training models with model.encoder.backbone.intrinsics_embed_loc='none'
python -m src.main +experiment=spfsplat/re10k wandb.mode=online wandb.name=re10k_nointrin model.encoder.backbone.intrinsics_embed_loc='none'

```

### Evaluation
#### Novel View Synthesis
```bash
# RealEstate10K (enable test.align_pose=true if using evaluation-time pose alignment)
python -m src.main +experiment=spfsplat/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./pretrained_weights/re10k.ckpt \
    test.save_image=true test.align_pose=false

# ACID (enable test.align_pose=true if using evaluation-time pose alignment)
python -m src.main +experiment=spfsplat/acid mode=test wandb.name=acid \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
  checkpointing.load=./pretrained_weights/acid.ckpt \
  test.save_image=false test.align_pose=false

# Multiple view evaluation on RealEstate10K
python -m src.main +experiment=spfsplat/re10k  mode=test wandb.name=re10k_10view \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    dataset.re10k.view_sampler.num_context_views=10 \
    checkpointing.load=./pretrained_weights/re10k_10view.ckpt
    test.save_image=false test.align_pose=false 

# RealEstate10K, evaluate on images without known intrinsics 
python -m src.main +experiment=spfsplat/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./pretrained_weights/re10k_nointrin.ckpt \
    model.encoder.backbone.intrinsics_embed_loc='none' \
    model.encoder.estimating_focal=true \
    test.save_image=true test.align_pose=false

# Evaluate on in-the-wild images, export .ply files, and render videos.  
# If camera intrinsics are available, please provide them in the code and use other checkpoints
python -m src.paper.validate_in_the_wild +experiment=spfsplat/re10k  wandb.name=re10k_iphone  \
  model.encoder.backbone.intrinsics_embed_loc='none' \
  model.encoder.estimating_focal=true \
  mode="test"  \
  checkpointing.load=models/re10k_nointrin.ckpt 

    
```



#### Pose Estimation
To evaluate the pose estimation performance, you can run the following command:
```bash
# RealEstate10K
python -m src.eval_pose +experiment=spfsplat/re10k +evaluation=eval_pose mode=test wandb.name=re10k \
  checkpointing.load=./pretrained_weights/re10k.ckpt \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json

# ACID
python -m src.eval_pose +experiment=spfsplat/acid +evaluation=eval_pose mode=test wandb.name=acid \
  checkpointing.load=./pretrained_weights/re10k.ckpt \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json

```
Note that here we show the evaluation using the model trained on RealEstate10K. You can replace the checkpoint path with other trained models.

## Camera Conventions
We follow the [pixelSplat](https://github.com/dcharatan/pixelsplat) camera system. The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).
The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Acknowledgements
This project is built upon these excellent repositories: [NoPoSplat](https://github.com/cvg/NoPoSplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [DUSt3R](https://github.com/naver/dust3r), and [CroCo](https://github.com/naver/croco). We thank the original authors for their excellent work.


## Citation

```
@article{huang2025spfsplat,
      title={No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views},
      author={Huang, Ranran and Mikolajczyk, Krystian},
      journal={arXiv preprint arXiv: 2508.01171},
      year={2025}
}
```