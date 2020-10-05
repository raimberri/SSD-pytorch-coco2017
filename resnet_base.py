#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# In[4]:


__all__ = ['ResnetBase', 'resnet_base']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# In[5]:


class ConvBnReluLayer(nn.Module):
    
    def __init__(self, inplanes, planes, kernel_size, padding, stride, bias = False):
        super(ConvBnReluLayer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# In[6]:


class ExtraLayers(nn.Module):
    
    def __init__(self, in_channel):
        super(ExtraLayers, self).__init__()
        self.convbnrelu9_1 = ConvBnReluLayer(in_channel, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu9_2 = ConvBnReluLayer(256, 256, kernel_size=3, padding=1, stride=2) 
        self.convbnrelu10_1 = ConvBnReluLayer(256, 128, kernel_size=1, padding=0, stride=1)
        self.convbnrelu10_2 = ConvBnReluLayer(128, 256, kernel_size=3, padding=1, stride=2)
        self.convbnrelu11_1 = ConvBnReluLayer(256, 128, kernel_size=1, padding=0, stride=1)
        self.convbnrelu11_2 = ConvBnReluLayer(128, 256, kernel_size=3, padding=0, stride=1)

        
    def forward(self, x):
        out9_1 = self.convbnrelu9_1(x)
        out9_2 = self.convbnrelu9_2(out9_1)
        out10_1 = self.convbnrelu10_1(out9_2)
        out10_2 = self.convbnrelu10_2(out10_1)
        out11_1 = self.convbnrelu11_1(out10_2)
        out11_2 = self.convbnrelu11_2(out11_1)
        return out9_2, out10_2, out11_2


# In[7]:


def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)


# In[8]:


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


# In[9]:


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        # set the padding=2 when dilation=2 to maintain the feature map size:


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


# In[10]:


class ResNetBase(nn.Module):
    
    def __init__(self, block, layers, width = 1, num_classes = 1000):
        super(ResNetBase, self).__init__()
        self.inplanes = 64
        widths = [int(round(ch * width)) for ch in [64, 128, 256, 512]]
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        # change stride = 2, dilation = 1 in ResNet to stride = 1, dilation = 2 for the final _make_layer
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2)
        
        self.inchannel = block.expansion * 512
        # remove the final avgpool and fc layers
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(widths[3] * block.expansion, num_classes)
        # add extra layers
        self.extra_layers = ExtraLayers(self.inchannel)
        self.smooth1 = nn.Conv2d(self.inchannel, 512, kernel_size = 3, stride = 1, padding = 1)
        
        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        out38x38 = x
        x = self.layer3(x)
        out19x19 = x
        x = self.layer4(x)
        out10x10 = x
        out10x10 = self.smooth1(out10x10)
  
        out5x5, out3x3, out1x1 = self.extra_layers(x)
        
        return out38x38, out19x19, out10x10, out5x5, out3x3, out1x1
    
def resnet_base(depth, width = 1, pretrained = False, **kwargs):
    """Constructs a ResNet base network model for SSD.
     Args:
        depth (int): choose 18, 34, 50, 101, 152
        width (float): widen factor for intermediate layers of resnet
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (depth not in [18, 34, 50, 101, 152]):
        raise ValueError('Choose 18, 34, 50, 101 or 152 for depth')  
    if ((width != 1) and pretrained):
        raise ValueError('Does not support pretrained models with width>1')
            
    name_dict = {18: 'resnet18', 34: 'resnet34', 50: 'resnet50', 101: 'resnet101', 152: 'resnet152'}
    layers_dict = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    block_dict = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
    model = ResNetBase(block_dict[depth], layers_dict[depth], width, **kwargs)  
    if ((width == 1) and pretrained):
        model.load_state_dict(model_zoo.load_url(model_urls[name_dict[depth]]))
    return model


# In[ ]:




