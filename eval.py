#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
from coco_dataset import *
from tqdm import tqdm
from pprint import PrettyPrinter


# In[2]:


pp = PrettyPrinter()

# Parameters
path2data = "./data/val2017"
path2json = "./data/annotations/instances_val2017.json"
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './Checkpoint_temp/checkpoint_epoch_67_ssd300.pth.tar'

# Load model to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']

model = model.to(device)

# eval mode
model.eval()

# Load test data
test_dataset = CocoDetectionForSSD(root=path2data, annFile=path2json, transform=None, target_transform=None, split='TEST')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn_SSD, num_workers = workers, pin_memory = True)


# In[3]:


def evaluate(test_loader, model):
    model.eval()
    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
   
    with torch.no_grad():
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc = 'Evaluating')):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            
            predicted_locs, predicted_scores = model(images)
            # Use paper's paramters
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores, min_score = 0.1, max_overlap = 0.45, top_k = 200)
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            #print('true_boxes:', boxes)
            #print('det_boxes:', det_boxes_batch)
        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
        
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)


# In[4]:


if __name__ == '__main__':
    evaluate(test_loader, model)


# In[2]:


det_boxes = [torch.tensor([[ 50.8456,  11.0575, 497.9483, 319.0857],[ 50.8456,  11.0575, 497.9483, 319.0857]])]
det_labels = [torch.tensor([1,1])]
det_scores = [torch.tensor([1,1])]
true_boxes = [torch.tensor([[ 67.9887, 155.5200, 276.2039, 240.4080],
                                      [ 11.3314,   7.7760, 498.5836, 322.7040]])]
true_labels = [torch.tensor([5, 1])]

calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)

