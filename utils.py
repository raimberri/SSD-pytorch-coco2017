#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import random
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
from shapely.geometry.point import Point
from shapely import affinity
import pygeos
from pygeos.measurement import bounds
from pygeos.measurement import area
from pygeos.set_operations import intersection

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COCO_LABELS = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

label_map = {k: v + 1 for v, k in enumerate(COCO_LABELS)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6','#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000','#ffd8b1', '#e6beff', '#808080', '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6','#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000','#ffd8b1', '#e6beff', '#808080', '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6','#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000','#ffd8b1', '#e6beff', '#808080', '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6','#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000','#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


# In[3]:


def decimate(tensor, m):
    """
    Decimate a tensor by 'm', for downsampling in converting FC layers to Conv layers
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch.arange(start=0, end=tensor.size(d), step=m[d]))
            
    return tensor


# In[4]:

def xy_to_cxcy(xy):
    """
    Convert (x_min, y_min, x_max, y_max) to (c_x, c_y, w, h)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)

def cxcy_to_xy(cxcy):
    """
    Convert (c_x, c_y, w, h) to (x_min, y_min, x_max, y_max)
    """
    return torch.cat([cxcy[:, :2] - cxcy[:, 2:] / 2, cxcy[:, :2] + cxcy[:, 2:] / 2], 1)

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encod bbox w.r.t the corresponding prior boxes in a manner stated in the paper.
    """
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10), torch.log(cxcy[:, 2:] + 1e-10 / priors_cxcy[:, 2:]) * 5], 1)

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode of the bbox encoding procedure.
    """
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2], torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)

def find_intersection(set_1, set_2):
    """
    return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2
    """
    # Pytorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


# In[5]:


def find_jaccard_overlap(set_1, set_2):
    """
    return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2
    """
    intersection = find_intersection(set_1, set_2)
    
    areas_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) # width * height
    areas_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])
    
    # Pytorch auto-broadcasts singleton dimensions
    union = areas_1.unsqueeze(1) + areas_2.unsqueeze(0) - intersection
    
    return intersection / (union + 1e-10)

 def create_ellipse(cx, cy, w, h):
    circ = Point((cx, cy)).buffer(1)
    ell = affinity.scale(circ, w, h)
    ell = ell.buffer(0)
    #ell = pygeos.io.from_shapely(ell)
    return ell
create_ellipse_vectorized = np.vectorize(create_ellipse)

def intersects(ells1, ells2):
    #return (ells1 & ells2).area
    return ells1.intersection(ells2).area
intersects_vectorized = np.vectorize(intersects)


def find_jaccard_overlap_ellipse(set_1, set_2):
    cxy1 = xy_to_cxcy(set_1).cpu()
    #cxy2 = xy_to_cxcy(set_2).cpu()
    ells1 = create_ellipse_vectorized(cxy1[:,0],cxy1[:,1],cxy1[:,2]/2,cxy1[:,3]/2)
    ells2 = set_2
    #ells2 = create_ellipse_vectorized(cxy2[:,0],cxy2[:,1],cxy2[:,2]/2,cxy2[:,3]/2)
    ells1 = pygeos.io.from_shapely(ells1)
    ells2 = pygeos.io.from_shapely(ells2)
    inter_area = np.zeros((ells1.shape[0], ells2.shape[0]), dtype = np.float32)
    # Rtree
    tree = pygeos.STRtree(ells1)
    idx = tree.query_bulk(ells2, predicate='intersects').tolist()
    inter_area[idx[1], idx[0]] = area(intersection(ells1[idx[1]], ells2[idx[0]]))
    '''idx = index.Index()
    for pos, cell in enumerate(ells1):
        idx.insert(pos, bounds(cell).tolist())

    for i, poly in enumerate(ells2):
        r = idx.intersection(bounds(poly).tolist())
        x = list(r)
        inter_area[x,i] = area(intersection(ells1[x], poly))'''
    #inter = intersection(ells1[:, np.newaxis], ells2)
    #inter_area = area(inter)
    b1_area = area(ells1)
    b2_area = area(ells2)
    union = b1_area[:,np.newaxis] + b2_area
    iou = inter_area / (union - inter_area + 1e-16)
    iou = torch.Tensor(iou).to(device)
    
    return iou

# In[6]:


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    return: expanded image, new bbox
    """
    # Calculate new dimensions
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)
    
    # Create a new background image with the filler
    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)
    
    # Place the original image at random position in the new background
    x_min = random.randint(0, new_w - original_w)
    x_max = x_min + original_w
    y_min = random.randint(0, new_h - original_h)
    y_max = y_min + original_h
    new_image[:, y_min:y_max, x_min:x_max] = image
    
    new_boxes = boxes + torch.FloatTensor([x_min, y_min, x_min, y_min]).unsqueeze(0)
    
    return new_image, new_boxes


# In[8]:


def random_crop(image, boxes, labels):
    """
    Perform a random crop operation in the manner stated in the paper.
    Some objects may dispear in the cropped image.
    return: cropped image, new bbox, new labels
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep trying until a successful crop is created
    while True:
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])
        
        # If not cropping
        if min_overlap is None:
            return image, boxes, labels
        
        # Try up to 50 times for this chosen overlapping ratio
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must larger than 0.3 times of original dimensions
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)
            
            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue
            
            # Cropped position coordinates
            x_min = random.randint(0, original_w - new_w)
            x_max = x_min + new_w
            y_min = random.randint(0, original_h - new_h)
            y_max = y_min + new_h
            crop = torch.FloatTensor([x_min, y_min, x_max, y_max])
            
            # Calculate Jaccard overlap between the cropped region and bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)
            overlap = overlap.squeeze(0)
            
            # If no bbox has a Jaccard overlap greater than the min_overlap ratio, try again
            if overlap.max().item() < min_overlap:
                continue
                
            # Crop image
            new_image = image[:, y_min:y_max, x_min:x_max]
            
            # Find centers of original bounding boxes
            bbox_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            
            # Find bbox whose center is in the cropped image
            centers_in_crop = (bbox_centers[:, 0] > x_min) * (bbox_centers[:, 0] < x_max) * (bbox_centers[:, 1] > y_min) * (bbox_centers[:, 1] < y_max)
            
            # If no bbox has its center in the cropped image, try again
            if not centers_in_crop.any():
                continue
                
            # Discard bbox that doesn't have its center in the cropped image
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            
            # Calculate new bbox coordinates
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]
            
            return new_image, new_boxes, new_labels


# In[9]:


def flip(image, boxes):
    """
    Flip images horizontally
    return: flipped image, new bbox
    """
    # Flip image
    new_image = FT.hflip(image)
    
    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - 1 - boxes[:, 0]
    new_boxes[:, 2] = image.width - 1 - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    
    return new_image, new_boxes


# In[11]:


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image.
    Can choose to return coordinates in a ralative/absolute manner.
    return: resized image, new bbox
    """
    # Resize image
    new_image = FT.resize(image, dims)
    
    # Resize bbox
    original_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / original_dims # relative coordinates
    
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
        
    return new_image, new_boxes


# In[13]:


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation and hue, each with a 50% chance, in random order.
    return: distorted image
    """
    new_image = image
    
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]
    
    random.shuffle(distortions)
    
    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
                
            # Apply distortion
            new_image = d(new_image, adjust_factor)
            
    return new_image


# In[14]:


def transform(image, boxes, labels, split):
    """
    Apply image transforms defined above.
    return: transformed image, new bbox, new labels
    """
    assert split in {'TRAIN', 'TEST'}
    # Mean and std used for pretrained VGG
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    new_image = image
    new_boxes = boxes
    new_labels = labels
    
    # Skip the following operations for testing/evaluation
    if split == 'TRAIN':
        # Photometric distortion
        new_image = photometric_distort(new_image)
        
        # ToTensor
        new_image = FT.to_tensor(new_image)
        
        # Expand with 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)
            
        # Randomly crop
        new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)
        
        # ToPIL
        new_image = FT.to_pil_image(new_image)
        
        # Flip with 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)
        
    # Resize and return relative bbox coordinates
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))
    
    # ToTensor
    new_image = FT.to_tensor(new_image)
    
    # Normalization
    new_image = FT.normalize(new_image, mean=mean, std=std)
    
    return new_image, new_boxes, new_labels

def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))
    
def save_checkpoint(epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_epoch_'+str(epoch)+'_ssd300.pth.tar'
    torch.save(state, filename)

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels):
    """
    Calculate mAP of model dectected objects.
    return: list of AP for each class adn mAP of all classes
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) # these are all lists of tensors
    n_classes = len(label_map)
    
    # Store all (true) objects in a single continuous tensor while record which image it is belonged to
    true_images = []
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device) # (n_objects)
    true_boxes = torch.cat(true_boxes, dim = 0) # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim = 0) # (n_objects)
    
    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)
    
    # Store all (detected) objects in a single continuous tensor while record which image it is belonged to
    det_images = []
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device) # (n_detections)
    det_boxes = torch.cat(det_boxes, dim = 0) # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim = 0) # (n_detections)
    det_scores = torch.cat(det_scores, dim = 0) # (n_detections)
    
    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)
    
    # Caculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype = torch.float) # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objeccts of this class
        true_images_of_this_class = true_images[true_labels == c] # (n_objects_of_this_class)
        true_boxes_of_this_class = true_boxes[true_labels == c] # (n_objects_of_this_class, 4)
        n_objects_of_this_class = true_boxes_of_this_class.size(0)    
        
        # Record which true object has already been 'detected'
        true_boxes_detected_of_this_class = torch.zeros((true_images_of_this_class.size(0)), dtype = torch.uint8).to(device) # (n_class_objects)
        
        # Extract only detections of this class
        det_images_of_this_class = det_images[det_labels == c] # (n_detections_of_this_class)
        det_boxes_of_this_class = det_boxes[det_labels == c] # (n_detections_of_this_class, 4)
        det_scores_of_this_class = det_scores[det_labels == c] # (n_detections_of_this_class)
        n_detections_of_this_class = det_boxes_of_this_class.size(0)
        
        if n_detections_of_this_class == 0:
            continue
        
        # Sort detections in decreasing order of confidence/scores
        det_scores_of_this_class, sort_ind = torch.sort(det_scores_of_this_class, dim = 0, descending = True)
        det_images_of_this_class = det_images_of_this_class[sort_ind]
        det_boxes_of_this_class = det_boxes_of_this_class[sort_ind]
        
        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_detections_of_this_class), dtype = torch.float).to(device)
        false_positives = torch.zeros((n_detections_of_this_class), dtype = torch.float).to(device)
        for d in range(n_detections_of_this_class):
            current_detection_box = det_boxes_of_this_class[d].unsqueeze(0) # (1, 4)
            current_image = det_images_of_this_class[d]
            
            # Find objects in the same image with this calss
            object_boxes = true_boxes_of_this_class[true_images_of_this_class == current_image] # (n_objects_of_this_class_in_this_image)
            # If no such object in this image, then the detection is a FP
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue
                
            # Find max overlap objects of this detection
            overlaps = find_jaccard_overlap(current_detection_box, object_boxes) # (1, n_objects_of_this_class_in_this_image)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim = 0)
            
            original_ind = torch.LongTensor(range(true_boxes_of_this_class.size(0)))[true_images_of_this_class == current_image][ind]
            
            # If there is a object paired with this detection, then this detection is a TP
            if max_overlap.item() > 0.5:
                # If this object has not been detected, this detection is a TP
                if true_boxes_detected_of_this_class[original_ind] == 0:
                    true_positives[d] = 1
                    true_boxes_detected_of_this_class[original_ind] = 1
                # Otherwise, this detection is a FP
                else:
                    false_positives[d] = 1
            # Otherwise, this detection is a FP
            else:
                false_positives[d] = 1
                
        # Compute cumulative precision and recall in decreasing orders for each detection
        cumul_TP = torch.cumsum(true_positives, dim = 0)
        cumul_FP = torch.cumsum(false_positives, dim = 0)
        cumul_precision = cumul_TP / (cumul_TP + cumul_FP + 1e-10)
        cumul_recall = cumul_TP / n_objects_of_this_class
        #print('cumul_TP:',cumul_TP)
        #print('cumul_FP:',cumul_FP)
        #print('cumul_precision:',cumul_precision)
        #print('cumul_recall:',cumul_recall)
        # 101-point interpolated AP
        recall_thresholds = torch.arange(start = 0, end = 1.1, step = 0.01).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype = torch.float).to(device)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0
        average_precisions[c - 1] = precisions.mean()


        
    mean_average_precision = average_precisions.mean().item()
    
    average_precisions = {rev_label_map[c + 1]:v for c, v in enumerate(average_precisions.tolist())}
    
    return average_precisions, mean_average_precision

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def clip_gradient(optimizer, grad_clip, c=1):
    for group in optimizer.param_groups:
        for param in group['params']:
            torch.nn.utils.clip_grad_norm_(param, grad_clip*c)
            '''if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param.grad.data, grad_clip*c)
                #param.grad.data.clamp_(-grad_clip, grad_clip)'''
            
def hard_negative_mining(loss, labels, neg_pos_ratio):
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask

def plot_grad_flow(named_parameters):
    '''
    Plug this function after loss.backwards() as
    "plot_grad_flow(model.named_parameters())"
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.figure(figsize = [15,10])
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
# In[ ]:




