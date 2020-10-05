#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model_resnet import SSD300ResNet, MultiBoxLoss
from coco_dataset import *
from utils import *


# In[2]:


path2data = "./data/train2017"
path2json = "./data/annotations/instances_train2017.json"

# Model parameters
n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Learning parameters
checkpoint = None # path to model checkpoint
batch_size = 32
iterations = 240000
workers = 4
print_freq = 200
lr = 1e-3
lr_decay_point = [160000, 200000]
lr_decay_ratio = 0.1
momentum = 0.9
weight_decay = 5e-4
grad_clip = True
clip = 5
cudnn.benchmark = True


# In[3]:


def train(train_loader, model, criterion, optimizer, epoch):
    model.train() # training mode enables dropout
    
    
    
    batch_time = AverageMeter() # foward prop + backward prop time
    data_time = AverageMeter() # data loading time
    losses = AverageMeter()
    losses_c = AverageMeter()
    losses_l = AverageMeter()
    start = time.time()
    
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
            
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        images, boxes, labels = data
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
           
        optimizer.zero_grad()
        # Forward prop
        predicted_locs, predicted_scores = model(images)
        
        # Loss
        loss_c, loss_l = criterion(predicted_locs, predicted_scores, boxes, labels) # scalar
        loss = loss_c + loss_l
        # Backward prop
        
        loss.backward()
        
        # Clip gradients, if necessary
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            #clip_gradient(optimizer, grad_clip, clip)
            
        # Update model
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        losses_c.update(loss_c.item(), images.size(0))
        losses_l.update(loss_l.item(), images.size(0))
        batch_time.update(time.time() - start)
        
        start = time.time()
        
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_c {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
                  'Loss_l {loss_l.val:.4f} ({loss_l.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time = batch_time, data_time = data_time, loss = losses, loss_c = losses_c, loss_l = losses_l))
        if i % (5 * print_freq) == 0:
            # Save new params
            new_state_dict = {}
            for key in model.state_dict():
                new_state_dict[key] = model.state_dict()[key].clone()
            # Compare params
            for key in old_state_dict:
                if (old_state_dict[key] == new_state_dict[key]).all():
                    print('Same in {}'.format(key))
            old_state_dict = new_state_dict
            plot_grad_flow(model.named_parameters())
    del predicted_locs, predicted_scores, images, boxes, labels
        


# In[4]:


def main():
    if checkpoint is None:
        start_epoch = 0
        model = SSD300ResNet(depth = 50, n_classes = n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases
        biases = []
        not_biases = []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params = [{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr = lr, momentum = momentum, weight_decay = weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoing from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    model = model.to(device)


    criterion = MultiBoxLoss(priors_cxcy = model.priors_cxcy).to(device)

    # Dataloaders
    train_dataset = CocoDetectionForSSD(root=path2data, annFile=path2json, transform=None, target_transform=None, split='TRAIN')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn_SSD, num_workers = workers, pin_memory = True)

    epochs = iterations // (len(train_dataset) // 32)
    lr_decay_point = [it // (len(train_dataset) // 32) for it in lr_decay_point]

    for epoch in range(start_epoch, epochs):
        if epoch in lr_decay_point:
            adjust_learning_rate(optimizer, lr_decay_ratio)
    
        train(train_loader = train_loader, model = model, criterion = criterion, optimizer = optimizer, epoch = epoch)
    
        save_checkpoint(epoch, model, optimizer)


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:




