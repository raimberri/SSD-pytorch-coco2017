#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
from resnet_base import resnet_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


class MultiBoxLayer(nn.Module):
    
    def __init__(self, n_classes):
        super(MultiBoxLayer, self).__init__()
        self.n_classes = n_classes
        
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        
        # Localization prediction convolutions
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size = 3, padding = 1)
        
        # Class prediction convolutions
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size = 3, padding = 1)
        
        self.init_conv2d()
        
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
    
    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)
        
        # Predict bboxes
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats) # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous() # (N, 38, 38, 16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4) # (N, 5776, 4)
        
        
        l_conv7 = self.loc_conv7(conv7_feats) # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous() # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4) # (N, 2166, 4)
        
        
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats) # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous() # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4) # (N, 600, 4)
        
        
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats) # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous() # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4) # (N, 150, 4)
        
        
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats) # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous() # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4) # (N, 36, 4)
        
        
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats) # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous() # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4) # (N, 4, 4)
       
        
        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats) # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous() # (N, 38, 38, 4 * n_classes)
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes) # (N, 5776, n_classes)
        
        c_conv7 = self.cl_conv7(conv7_feats) # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous() # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes) # (N, 2166, n_classes)
        
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats) # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous() # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes) # (N, 600, n_classes)
        
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats) # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous() # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes) #(N, 150, n_classes)
        
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats) # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous() # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes) # (N, 36, n_classes)
        
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats) # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous() # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes) # (N, 4, n_classes)
        
        # Concatenate all boxes
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim = 1) # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim = 1) # (N, 8732, n_classes)
        
        return locs, classes_scores


# In[5]:


class SSD300ResNet(nn.Module):
    input_size = 300
    
    
    def __init__(self, depth, width = 1, n_classes=81):
        super(SSD300ResNet, self).__init__()
        self.base_network = resnet_base(depth, width)
        self.multibox = MultiBoxLayer(n_classes)
        self.n_classes = n_classes
        if depth == 18 or depth == 34:
            self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 128, 1, 1))
            nn.init.constant_(self.rescale_factors, 20)
        else:
            # Rescale lower-level features
            self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
            nn.init.constant_(self.rescale_factors, 20)
        # Default boxes
        self.priors_cxcy = self.create_prior_boxes()
        
    def forward(self, image):
        out38x38, out19x19, out10x10, out5x5, out3x3, out1x1 = self.base_network(image)
        # Rescale conv4_3_feats using L2 norm
        norm = out38x38.pow(2).sum(dim = 1, keepdim = True).sqrt() # (N, 1, 38, 38)
        out38x38 = out38x38 / norm # (N, 512, 38, 38)
        out38x38 = out38x38 * self.rescale_factors # (N, 512, 38, 38)
        loc_preds, conf_preds = self.multibox(out38x38, out19x19, out10x10, out5x5, out3x3, out1x1)
        return loc_preds, conf_preds
    
    def create_prior_boxes(self):
        """
        Create 8732 default boxes for SSD
        return: prior boxes in center-size form, (8732, 4) tensor
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}
        # COCO standard in paper
        obj_scales = {'conv4_3': 0.07,
                     'conv7': 0.15,
                     'conv8_2': 0.3375,
                     'conv9_2': 0.525,
                     'conv10_2': 0.7125,
                     'conv11_2': 0.9}
        
        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                     'conv7': [1., 2., 3., 0.5, .333],
                     'conv8_2': [1., 2., 3., 0.5, .333],
                     'conv9_2': [1., 2., 3., 0.5, .333],
                     'conv10_2': [1., 2., 0.5],
                     'conv11_2': [1., 2., 0.5]}
        
        fmaps = list(fmap_dims.keys())
        
        
        prior_boxes = []
        
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    
                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])                        
                        # For aspect ratio == 1, add a scale s_k' = sqrt(s_k * s_{k+1})
                        if ratio == 1:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # The last feature map does not have next feature map, thus set scale == 1(> 0.9)   
                            except IndexError:
                                additonal_scale = 1
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
                            
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0.0, 1.0) # (8732, 4)
        
        return prior_boxes
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of the SSD300) to detect objects.
        
        For each class, perform Non-Maximum Suppression on boxes that are above a minimum threshold.
        
        min_score: minmum threshold for reserve the box for a certain class
        max_overlap: maximum overlap between two boxes that the one with a lower score won't be suppressed
        top_k: if there are too many detected objects, keep the top k results
        return: detections(bboxes, labels, scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        with torch.no_grad():
            predicted_scores = F.softmax(predicted_scores, dim = 2) # (N, 8732, n_classes)
        
        # Lists to store final results
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        
        for i in range(batch_size):
            # Decode object coordinates from g to xy(relative coordinates)
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            # Lists to store results for one image
            image_boxes = []
            image_labels = []
            image_scores = []
            
            max_scores, best_label = predicted_scores[i].max(dim = 1) # (8732)
            
            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only results are above the minimum score
                class_scores = predicted_scores[i][:, c] # (8732)
                score_above_min_score = class_scores > min_score # [0, 1, ... 0] tensor for indexing
                n_above_min_score = score_above_min_score.sum().item()
                #print('n_above_min_score:',n_above_min_score)
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score] # (n_qualified)
                class_decoded_locs = decoded_locs[score_above_min_score] # (n_qualified, 4)
                
                # Sort qualified results
                class_scores, sort_ind = class_scores.sort(dim = 0, descending = True)
                class_decoded_locs = class_decoded_locs[sort_ind]
                
                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs) # (n_qualified, n_qualified)

                # Non-Maximum Suppression
                
                # Use a torch.uint8 tensor to indicate which predicted box is suppressed
                # 1 implies suppressed
                suppress = torch.zeros((n_above_min_score)).bool().to(device) # (n_qualified)
                
                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue
                        
                    # Suppress boxes whose overlaps are greater than maximum overlap
                    suppress = suppress | (overlap[box] > max_overlap)
                    #overlap_above_max = (overlap[box] > max_overlap).bool()
                    #suppress = torch.max(suppress, overlap_above_max)
                    suppress[box] = 0 # as the overlap of iteself is 1
                    
                # Store the unsuppressed boxes for this class
                
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])
                
            # If no object is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))
                
            # Concatenate results list to tensors
            image_boxes = torch.cat(image_boxes, dim = 0) # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim = 0) # (n_objects)
            image_scores = torch.cat(image_scores, dim = 0) # (n_objects)
            n_objects = image_scores.size(0)
            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim = 0, descending = True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]
                
            # Store the final results
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
            
        return all_images_boxes, all_images_labels, all_images_scores


# In[6]:


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss is the combination of localization loss and confidence loss
    """
    
    def __init__(self, priors_cxcy, threshold = 0.5, neg_pos_ratio = 3, alpha = 1.0):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        self.smooth_l1 = nn.SmoothL1Loss(reduction = 'sum')
        self.cross_entropy = nn.CrossEntropyLoss(reduction = 'sum')
        
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        boxes: groundtruth bbox, a list of N tensors
        labels: groundtruth labels, a list of N tensors
        return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype = torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype = torch.long).to(device)
        
        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy) # (n_objects, 8732)
            
            # For each prior, find the object with the maximum overlap
            overlap_for_each_prior, object_index_for_each_prior = overlap.max(dim = 0) # (8732)
            
            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_index_for_each_object = overlap.max(dim = 1) # (N_object)
            
            # Then, assign each object to the corresponding maximum-overlap-prior(This fixes 1.)
            object_index_for_each_prior[prior_index_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            
            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_index_for_each_object] = 1
            
            # Labels for each prior
            label_for_each_prior = labels[i][object_index_for_each_prior] # (8732)
            # Set priors whose overlaps are less than the threshold to background
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            
            # Store
            true_classes[i] = label_for_each_prior # (8732)
            
            # Encode groundtruth bbox coordinates from center-size to g
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_index_for_each_prior]), self.priors_cxcy) # (8732, 4)
            
        # Indentify priors that are positive(non-background)
        positive_priors = true_classes != 0 # (N, 8732)
        
        with torch.no_grad():
            loss = -F.log_softmax(predicted_scores, dim = 2)[:, :, 0]
            mask = hard_negative_mining(loss, true_classes, self.neg_pos_ratio)
            
        confidence = predicted_scores[mask, :]
        classification_loss = self.cross_entropy(confidence.view(-1, n_classes), true_classes[mask])
        
        pos_mask = true_classes > 0
        predicted_locations = predicted_locs[pos_mask, :].view(-1, 4)
        gt_locations = true_locs[pos_mask, :].view(-1, 4)
        smooth_l1_loss = self.smooth_l1(predicted_locations, gt_locations)
        num_pos = gt_locations.size(0)
        return classification_loss / num_pos, smooth_l1_loss / num_pos
        '''# LOCALIZATION LOSS
        # Localization loss is computed only on positive priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]) # scalar
        
        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)
        
        # CONFIDENCE LOSS
        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance
        
        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim = 1) # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives # (N)
        
        # First, find the loss of all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1)) # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors) # (N, 8732)
        
        # Find loss of positive priors
        conf_loss_pos = conf_loss_all[positive_priors] # (sum(n_positive))
        
        # Find loss of hard-negative priors
        conf_loss_neg = conf_loss_all.clone() # (N, 8732)
        conf_loss_neg[positive_priors] = 0 # (N, 8732) ignore positive priors' loss
        conf_loss_neg, _ = conf_loss_neg.sort(dim = 1, descending = True) # (N, 8732)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device) # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1) # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives] # (sum(n_hard_negatives))
        
        # As in the paper, averaged over positive priors only
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()
        loc_loss = loc_loss / n_positives.sum().float()
        # TOTAL LOSS
        return conf_loss, self.alpha * loc_loss'''


# In[ ]:




