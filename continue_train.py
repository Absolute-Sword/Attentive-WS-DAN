#!/usr/bin/env python
# coding: utf-8

# In[1]:

from torch.utils.data import DataLoader, Dataset


import os
from model.utils.center_tool import generate_map
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import data.voc_dataset as dv
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.config import opt
from data.dataset import TestDataset, inverse_normalize, Dataset
from data.cutmix_dataset import Cutmix_dataset
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
# from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import torch
import resource
import logging
import time
import warnings
import numpy as np


# In[3]:


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
# matplotlib.use('TkAgg')

    
def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break 
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


# In[4]:


dataset = Cutmix_dataset(path="cutmix_data/fusion_cutmix_25_0.1_data_pic-per-image",gzip_flag=True)    
# dataset = Cutmix_dataset()
cutmix_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=opt.num_workers)







train_dataset = Dataset(opt)
dataloader = data_.DataLoader(train_dataset,                               batch_size=1,                               shuffle=True,                               pin_memory=True,
                              num_workers=opt.num_workers)


# In[8]:


testset = TestDataset(opt)
    
test_dataloader = data_.DataLoader(testset,
                                    batch_size=1,
                                    num_workers=opt.test_num_workers,
                                    shuffle=True, \
#                                     pin_memory=True
                                    )


# In[9]:


faster_rcnn = FasterRCNNVGG16().cuda()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()


# In[10]:


run_name = 'cutmix_data_1e-3_0.1_data'
# run_name = "text.txt"
logging.basicConfig(
        filename=os.path.join("./log", f"{run_name}_{time.strftime('%Y-%m-%d-%H-%M')}-log.txt"),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
warnings.filterwarnings("ignore")


# In[11]:

if opt.load_path:
    trainer.load(opt.load_path)
    print('load pretrained mode`l from %s' % opt.load_path)


# In[43]:


# trainer.update_lr(1e-4)


# In[ ]:


best_map = 0
lr_ = opt.lr
faster_rcnn.train()
plot_flag = False
cutmix_flag = False

for epoch in range(opt.epoch):
    trainer.reset_meters()    
    
    loss_history = []
    loss_batch_history = []
    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(cutmix_dataloader)):        
        scale = at.scalar(scale)
        img, bbox, labels = img.cuda().squeeze(0).float(), bbox_.cuda().squeeze(0), label_.cuda()
        copy_cache = [img, bbox, labels]
        paste_cache = [None, None, None]        
        paste_scale = 1
        losses, info, imgs, bboxes, labels = trainer.train_step(img, scale, paste_scale, *copy_cache, *paste_cache, cutmix_flag, plot_flag)
        loss_history.append(losses.total_loss.item()) 
        
        if (ii + 2) % opt.plot_every == 0:    
            logging.info(f"[Batch: {epoch}/Iter {ii + 1}] cutmix training loss: {np.mean(loss_history):.2f}")      
            loss_batch_history.append(np.mean(loss_history))
#    plt.plot(range(len(loss_batch_history)), loss_batch_history) 
#    plt.savefig(f"./loss_history/epoch{epoch}_cutmix.png")
#    plt.show()       
    
#    trainer.update_lr(lr*2)    
    loss_history = []
    loss_batch_history = []
    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        scale = at.scalar(scale)
        img, bbox, labels = img.cuda().float(), bbox_.cuda(), label_.cuda()
        
        copy_cache = [img, bbox, labels]
        paste_cache = [None, None, None]
        paste_scale = 1
        losses, info, imgs, bboxes, labels = trainer.train_step(img, scale, paste_scale, *copy_cache, *paste_cache, cutmix_flag, plot_flag, device='cuda')
        loss_history.append(losses.total_loss.item())        
        if (ii + 1) % opt.plot_every == 0:                
            logging.info(f"[Batch: {epoch}/Iter {ii + 1}] orin training loss: {np.mean(loss_history):.2f}")                            
            loss_batch_history.append(np.mean(loss_history))
            
#    plt.plot(range(len(loss_batch_history)), loss_batch_history) 
#    plt.savefig(f"./loss_history/epoch{epoch}_origin.png")
#     plt.show()
#             imgs = inverse_normalize(at.tonumpy(imgs.squeeze())) / 255
#             plt.figure(figsize=(8, 8))
#             plt.imshow(imgs.transpose(1, 2, 0))    
#             if not isinstance(bboxes, np.ndarray) and not isinstance(bboxes, torch.Tensor):
#                 input_bboxes = np.array(input_bboxes)        
#             input_bboxes = bboxes.reshape(-1, 4)
#             w = input_bboxes[:, 3] - input_bboxes[:, 1]
#             h = input_bboxes[:, 2] - input_bboxes[:, 0]    
#             for i in range(input_bboxes.shape[0]):                                                             
#                 plt.gca().add_patch(Rectangle(input_bboxes[i][[1, 0]],w[i], h[i], fill=False,edgecolor='r'))              
#                 plt.text(input_bboxes[i][1], input_bboxes[i][0], dv.VOC_BBOX_LABEL_NAMES[labels.reshape(-1, len(input_bboxes))[0][i]])            
#             plt.axis("off")     
#             plt.show()            
#         break
#     print("*"*100)
    
#         imgs = inverse_normalize(at.tonumpy(imgs.squeeze())) / 255
#         plt.figure(figsize=(8, 8))
#         plt.imshow(imgs.transpose(1, 2, 0))    
#         if not isinstance(bboxes, np.ndarray) and not isinstance(bboxes, torch.Tensor):
#             input_bboxes = np.array(input_bboxes)        
#         input_bboxes = bboxes.reshape(-1, 4)
#         w = input_bboxes[:, 3] - input_bboxes[:, 1]
#         h = input_bboxes[:, 2] - input_bboxes[:, 0]    
#         for i in range(input_bboxes.shape[0]):                                                             
#             plt.gca().add_patch(Rectangle(input_bboxes[i][[1, 0]],w[i], h[i], fill=False,edgecolor='r'))              
#             plt.text(input_bboxes[i][1], input_bboxes[i][0], dv.VOC_BBOX_LABEL_NAMES[labels.reshape(-1, len(input_bboxes))[0][i]])            
#         plt.axis("off")     
#         plt.show()    
#         break
    
     
    
    lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
    logging.info(f"[Batch: {epoch}] training loss: {np.mean(loss_history):.2f} lr: {lr_}")
    if (epoch+1) % 1 == 0:        
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        logging.info(f"[Batch: {epoch}] eval loss: {eval_result['map']:.4f}")
        if eval_result['map'] > best_map:            
            best_map = eval_result['map']
            if best_map > 0.68:
                best_path = trainer.save(best_map=best_map)   

    if epoch==9:
#       pass
        trainer.faster_rcnn.scale_lr(opt.lr_decay)
        lr_ = lr_ * opt.lr_decay








