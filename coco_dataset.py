#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
from utils import transform


# In[ ]:

HOME = os.path.expanduser("~")
# Modify the coco dataset path here
COCO_ROOT = os.path.join(HOME, 'Documents/SSD-pytorch/data/')

class AnnotationTransform(object):
    """Transform a COCO annotation into a list of bbox parameters and category indices
    Args:
        output_size(tuple or int): Desired image crop size. Used to modified bbox parameters.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, target):
        if isinstance(self.output_size, int):
            new_width, new_height = self.output_size, self.output_size
        else:
            new_width, new_height = self.output_size
            
        new_width, new_height = int(new_width), int(new_height)
        scale = np.array([new_width, new_height, new_width, new_height])
        labels = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[0] += bbox[2] * 0.5
                bbox[1] += bbox[3] * 0.5
                category_idx = obj['category_id'] - 1
                final_bbox = list(np.array(bbox) / scale)
                final_bbox.append(category_idx)
                labels += [final_bbox]
            else:
                print("no bbox exists!")
                
        return labels # [[cx, cy, w, h, category_id], ...]
        


# In[ ]:

def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map

class CocoDetectionForSSD(VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, split='TRAIN'):
        super(CocoDetectionForSSD, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.split = split.upper()
        self.label_map = get_label_map(os.path.join(COCO_ROOT, 'coco_labels.txt'))
        
        assert self.split in {'TRAIN', 'TEST'}
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        assert os.path.exists(os.path.join(self.root, path)), 'Image path does not exist: {}'.format(path)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        
        boxes = []
        labels = []
        
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                category_idx = self.label_map[obj['category_id']]
                labels += [category_idx]
                boxes += [bbox]     
            else:
                print("no bbox exists!")
                
        if not boxes:
            
            bbox = [0.,0.,0.,0.]
            boxes += [bbox]
            labels += [0]
        
        boxes = np.array(boxes)
        labels = np.array(labels)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        img, bbox, labels = transform(img, boxes, labels, split=self.split)
        bbox = bbox.clamp_(min = 0.0, max = 1.0)
        
        return img, bbox, labels
        
    
    def __len__(self):
        return len(self.ids)
        
def collate_fn_SSD(batch):
    images = []
    bbox = []
    labels = []
    for sample in batch:
        images.append(sample[0])
        bbox.append(sample[1])
        labels.append(sample[2])
    return torch.stack(images, dim=0), bbox, labels
