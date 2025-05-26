## BARIS-ERA: Boundary-Aware Refinement with Environmental Robust Adapter Tuning for Underwater Instance Segmentation
This repository is the official PyTorch implementation of BARIS-ERA: Boundary-Aware Refinement with Environmental Robust Adapter Tuning for Underwater Instance Segmentation.


## Getting Started
```bash
git clone https://github.com/PANpinchi/BARIS-ERA.git

cd BARIS-ERA
```

## Installation and Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
conda create -n baris_era python=3.10

conda activate baris_era

source install_environment.sh
```
or manually execute the following command:
```bash
# CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

pip install -v -e .

pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

pip install opencv-contrib-python
pip install terminaltables
pip install pycocotools
pip install scikit-learn
pip install numpy==1.23.5
pip install gdown
pip install mmcls
pip install yapf==0.40.1
```

Run the commands below to download the pre-trained model:
```bash
mkdir pretrained

cd pretrained

gdown --id 1-nK4MYPiW5bB8wDHbIXzLimRkLLpek6x

gdown --id 1_MxeMnI11CuvWHGEvud7COMwsPyVeNNv

cd ..
```
Note: `*.pth` files should be placed in the `/pretrained` folder.


## Demo
Run the commands below to perform a pretrained model on images.
```bash
python vis_infer.py
```

## Citation
If you use this code, please cite the following:
```bibtex
@misc{pan2024_hmropt,
    title  = {BARIS-ERA: Boundary-Aware Refinement with Environmental Robust Adapter Tuning for Underwater Instance Segmentation},
    author = {Pin-Chi Pan and Soo-Chang Pei},
    url    = {https://github.com/PANpinchi/BARIS-ERA},
    year   = {2025}
}
```
