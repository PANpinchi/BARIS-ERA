#!/bin/bash

echo "✅ Installing PyTorch + CUDA Toolkit via conda..."
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

echo "✅ Installing current project in editable mode..."
pip install -v -e .

echo "✅ Installing mmcv-full..."
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

echo "✅ Installing additional Python dependencies..."
pip install opencv-contrib-python
pip install terminaltables
pip install pycocotools
pip install scikit-learn
pip install numpy==1.23.5
pip install gdown
pip install mmcls
pip install yapf==0.40.1

echo "🎉 All packages installed successfully!"
