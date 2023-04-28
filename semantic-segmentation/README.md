# LargeKernel3D: Scaling up Kernels in 3D Sparse CNNs (CVPR 2023)

This is the semantic segmentation part of LargeKernel3D, we follow the settings in [Stratified-Transformer](https://github.com/dvlab-research/Stratified-Transformer) for training and testing. 

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Datasets Preparation
Please refer to https://github.com/dvlab-research/PointGroup for the ScanNetv2 preprocessing. Then change the `data_root` entry in the config.yaml configuration file accordingly.

### 3. Training
```
python3 train.py --config config.yaml
```

### 4. Testing
For testing, first change the `model_path`, `save_folder` and `data_root_val` (if applicable) accordingly. Then, run the following command. 
```
python3 test.py --config config.yaml
```

Note that if you installed [spconv-plus](https://github.com/dvlab-research/spconv-plus), you can use `spatialgroupconvv2` in the network. Otherwise, `spatialgroupconv` is used by default. Note that no speedup would be available without [spconv-plus](https://github.com/dvlab-research/spconv-plus).

### Experimental results

| ScanNetv2 Semantic Segmentation |  Set | mIoU |
|-----------------------------------------------------------------------------------|:----:|:----:|
| LargeKernel3D | val | 73.5 |
| LargeKernel3D | test | 73.9 |
