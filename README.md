<div align='center'>

<h2><a href="https://ieeexplore.ieee.org/abstract/document/10659158">Uni-to-Multi Modal Knowledge Distillation for Bidirectional LiDAR-Camera Semantic Segmentation (TPAMI2024)</a></h2>

Tianfang Sun<sup>1</sup>, Zhizhong Zhang<sup>1</sup>, Xin Tan<sup>1</sup>, Yong Peng<sup>3</sup>, Yanyun Qu<sup>2</sup>, Yuan Xie<sup>1</sup>
<br>
 
<sup>1</sup>ECNU, <sup>2</sup>XMU, <sup>3</sup>CSU
 
</div>

# Installation

For easy installation, we recommend using [conda](https://www.anaconda.com/):

```shell
conda create -n u2mkd python=3.9
conda activate u2mkd
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip3 install numba tensorboard
# to support nuscenes
pip3 install nuscenes-devkit
```

Our method is based on [torchpack](https://github.com/zhijian-liu/torchpack) and [torchsparse](https://github.com/mit-han-lab/torchsparse). To install torchpack, we recommend to firstly install openmpi and mpi4py.

```shell
conda install -c conda-forge mpi4py openmpi
```

Install torchpack

```shell
pip install git+https://github.com/zhijian-liu/torchpack.git
```

Before installing torchsparse, install [Google Sparse Hash](https://github.com/sparsehash/sparsehash) library first.

```shell
sudo apt install libsparsehash-dev
```

Then install torchsparse (v1.4.0) by

```shell
pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

to support SphereFormer, for more details, please refer to [SphereFormer](https://github.com/dvlab-research/SphereFormer)

```shell
pip install torch_scatter==2.1.2
pip install torch_geometric==1.7.2
pip install spconv-cu114==2.3.6
pip install torch_sparse==0.6.18 cumm-cu114==0.4.11 torch_cluster==1.6.3
pip install timm termcolor tensorboardX
# Install sptr
cd third_party/SparseTransformer && python setup.py install
```

# Model Preparation

Please download ImageNet pretrained weight for SwiftNet from [Google Drive](https://drive.google.com/file/d/17Z5fZMcaSkpDdcm6jDBaXBFSo1eDxAsD/view?usp=sharing) or [BaiduDisk](https://pan.baidu.com/s/17Wn_zj69v1_QdjAP1v7eMw?pwd=063m).

# Data Preparation

## Download Datasets
Please download the datasets following the official instruction. The official websites of each dataset are listed as following: [nuScenes_lidarseg](https://www.nuscenes.org/nuscenes#download), [SemanticKITTI](http://www.semantic-kitti.org/dataset.html), [Waymo open](https://waymo.com/open/).
The color images of SemanticKITTI datasets can be downloaded from [KITTI-odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset.

## Generate Instance Augmentation Database
```bash
# nuScenes_lidarseg
python3 prepare_nusc_inst_database.py
# SemanticKITTI
python3 prepare_semkitti_inst_database.py
# Waymo Open Set
python3 prepare_waymo_inst_database.py
```

# Training

## nuScenes_lidarseg

1. Run the following command to train uni-modal teacher model. (*e.g.* SphereFormer)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python3 train_spformer.py configs/nuscenes/train/spformer.yaml --run-dir runs/nusc/spvcnn_spformer_cr2p0_multisweeps4_ep25_seed123
```

2. Modify the ``teacher_pretrain`` in ``configs/nuscenes/train/spformer_tsd_full_ours_star.yaml`` to the path of uni-modal teacher model trained in Step 1.

```yaml
teacher_pretrain: /data2/stf/codes/lifusion/runs/nusc_rq/spvcnn_spformer_cr2p0_multisweeps4_ep25_seed123/checkpoints/max-iou-val-vox.pt
```

3. Run the following command to train cross-modal student model.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python3 train_lc_nusc_tsd_full.py configs/nuscenes/train/spformer_tsd_full_ours_star.yaml --run-dir runs/nusc/spformer_swiftnet18_cr2p0_tsd_multisweeps4_ep25_seed123
```

# Acknowledgement
This repo is built upon [torchpack](https://github.com/zhijian-liu/torchpack), [torchsparse](https://github.com/mit-han-lab/torchsparse), [SphereFormer](https://github.com/dvlab-research/SphereFormer), [SwiftNet](https://github.com/orsic/swiftnet).

# Citation
If you find this repo useful, please consider citing our paper:
```bibtex
@ARTICLE{sun2024u2mkd,
  author={Sun, Tianfang and Zhang, Zhizhong and Tan, Xin and Peng, Yong and Qu, Yanyun and Xie, Yuan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Uni-to-Multi Modal Knowledge Distillation for Bidirectional LiDAR-Camera Semantic Segmentation}, 
  year={2024},
  volume={46},
  number={12},
  pages={11059-11072},
  doi={10.1109/TPAMI.2024.3451658}}
```