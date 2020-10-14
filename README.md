# SSD-pytorch-coco2017
SSD pytorch implementation for RA application.  
Still have some structure issues, only have 1.8% mAP on VGG-based and 0.5% mAP on Resnet50-based networks.  
**Updated:**  
Add oval anchors modification in util.py and model.py
Here is some information of working environments and final results:
## Environments
1. Ubuntu 20.04
2. Python 3.8
3. PyTorch 1.6.0
## Results
|    | VGG126 | Resnet50 |
|:-:|:-:|:-:|
|mAP | 1.8%   |     0.5% |
## Reference
Code modified from:  
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection  
https://github.com/amdegroot/ssd.pytorch  
https://github.com/lufficc/SSD
