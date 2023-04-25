# LargeKernel3D: Scaling up Kernels in 3D Sparse CNNs (CVPR 2023)

This is the object detection part of LargeKernel3D, we directly follow the settings in [FocalsConv](https://github.com/dvlab-research/FocalsConv/tree/master/CenterPoint), based on the CenterPoint detector. 

You can directly put the configs and backbone network files into it. And then follow the instruction in [FocalsConv-CenterPoint](https://github.com/dvlab-research/FocalsConv/tree/master/CenterPoint) for training and testing.
Both LiDAR-only and multi-modal settings are supported.

Note that if you installed [spconv-plus](https://github.com/dvlab-research/spconv-plus), you can use `spatialgroupconvv2` in the network. Otherwise, `spatialgroupconv` is used by default.

### Experimental results

| nuScenes Object Detection                                                                                                    |      Set       | mAP  | NDS  |            Download            |
|------------------------------------------------------------------------------------------------------------------------------|:--------------:|:----:|:----:|:------------------------------:|
| [LargeKernel3D](object-detection/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_largekernel3d_tiny.py)   |      val       | 63.3 | 69.1 |        [Pre-trained](https://drive.google.com/file/d/1qDCareDEyzElFMH0iPuMYkMVozI8qSGQ/view?usp=share_link)         |
| [LargeKernel3D](object-detection/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_largekernel3d_multimodal.py) |      test      | 65.4 | 70.6 | [Pre-trained](https://drive.google.com/file/d/1Cipmcq5PFyxObWkJPG9LPUNVnYsrlYBH/view?usp=share_link) [Submission](https://drive.google.com/file/d/1y2Km6rCb7PFBoe458cYL-H4Jh3yDeBe1/view?usp=share_link) |
| +test aug  |      test      | 68.7 | 72.8 |         [Submission](https://drive.google.com/file/d/15gYXgwE6XSIJhrnEFFvwKHXdPv9rK_PM/view?usp=share_link)         |
| [LargeKernel3D-F](object-detection/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_largekernel3d_multimodal.py) | test |  -   |  -   | [Pre-trained](https://drive.google.com/file/d/1MDSOGEtV0BZ_GCWDiedyLe9h1pi-lnnV/view?usp=share_link) |
| +test aug  |      test      | 71.1 | 74.2 |         [Submission](https://drive.google.com/file/d/1eQkQRA7YPAn6DuEh6oUA_VOv6vvs1csF/view?usp=share_link)         |
