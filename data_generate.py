

import os
from model.utils.center_tool import generate_map
import matplotlib.pyplot as plt
import matplotlib
from utils import array_tool as at
from matplotlib.patches import Rectangle
import data.voc_dataset as dv
from tqdm import tqdm
# import tqdm.notebook.tqdm as tqdm
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
import torch
import resource
import logging
import time
import warnings
import numpy as np
import pickle
import gzip


# In[3]:


# def generate():
#     opt._parse(kwargs)

dataset = Dataset(opt)    
dataloader = data_.DataLoader(dataset,                               batch_size=1,                               shuffle=False,                               pin_memory=True,
                              num_workers=1)


# In[4]:


device='cuda'


# In[5]:


faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).to(device)


# In[6]:


load_path = 'checkpoints/fasterrcnn_06271437_0.6936766760275197'
if load_path:
    trainer.load(load_path, device=device)
    print('load pretrained model from %s' % load_path)


# In[7]:


count_threshold = 25
type=f'cutmix_{count_threshold}_0.1_data_pic-per-image'


# In[8]:


path = os.path.join("cutmix_data",type)
if not os.path.exists(path):
    os.mkdir(path)    


# In[9]:


saved = []
for i in open("cutmix_data/data2.txt").readlines():  
    saved.append(int(i.split("_")[1].strip()))    
print(saved)


# In[ ]:





# In[21]:


for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
    if ii not in saved:
        continue
#     if ii < 345:
#         continue
#     if ii<1764: # 到1764，1764刚进行
#         continue
    index = list(range(len(dataset))) 
    np.random.shuffle(index)
    scale = at.scalar(scale)    
    data_name=f"data_{ii}"
    
    data = {}
    data["imgs"]=[]
    data["bboxes"]=[]
    data["labels"]=[]
    data["scales"]=[]  
    data["area"]=[]
    img, bbox, labels = img.to(device).float(), bbox_.to(device), label_.to(device)
    bbox = bbox.reshape(-1, 4)            
    labels = labels.reshape(-1, 1)   
    count = 0
    copy_cache = [img, bbox, labels] 
#     print(bbox.shape, labels.shape)
    for ite, jj in tqdm(enumerate(index)):
        torch.cuda.empty_cache()
        paste_img, paste_bboxes, paste_labels, paste_difficult = dataset.db.get_example(jj)
        paste_img, paste_bboxes, paste_labels, paste_scale = dataset.tsf((paste_img, paste_bboxes, paste_labels))
#         print('paste',paste_bboxes.shape, paste_labels.shape)
        if scale == paste_scale:            
#         pasteimg, paste_bboxes, paste_labels = paste_img.cuda().float(), paste_bboxes.cuda(), paste_labels.cuda()
            paste_bboxes = np.array(paste_bboxes).reshape(-1, 4)            
            paste_labels = np.array(paste_labels).reshape(-1, 1)
            paste_cache = [paste_img, paste_bboxes, paste_labels]
            imgs, bboxes, labels, _, info = trainer.cutmix_process(img, scale, paste_scale, *copy_cache, *paste_cache, 0.6, 0.7, device=device)
            if info["use_cutmix"] == 1:               
                data["imgs"].append(imgs.detach().cpu().numpy())
                data["bboxes"].append(bboxes.detach().cpu().numpy())
                data["labels"].append(labels.detach().cpu().numpy())
                data["scales"].append(scale)
                data["area"].append(info["area"])
                count += 1
#                 del imgs
#                 del bboxes
#                 del labels

#                 imgs = inverse_normalize(at.tonumpy(imgs.squeeze())) / 255
#                 plt.figure(figsize=(8, 8))
#                 plt.imshow(imgs.transpose(1, 2, 0))    
#                 if not isinstance(bboxes, np.ndarray) and not isinstance(bboxes, torch.Tensor):
#                     input_bboxes = np.array(input_bboxes)        
#                 input_bboxes = bboxes.reshape(-1, 4)
#                 w = input_bboxes[:, 3] - input_bboxes[:, 1]
#                 h = input_bboxes[:, 2] - input_bboxes[:, 0]  

#                 for i in range(input_bboxes.shape[0]):                                                             
#                     plt.gca().add_patch(Rectangle(input_bboxes[i][[1, 0]],w[i], h[i], fill=False,edgecolor='r'))              
#                     plt.text(input_bboxes[i][1], input_bboxes[i][0], dv.VOC_BBOX_LABEL_NAMES[labels.reshape(-1, len(input_bboxes))[0][i]])            
#                 plt.axis("off")                 
#                 plt.show()

        else:
#             print("pass")
            pass
        if count >= count_threshold or (ite==int(count_threshold/2) and count == 0) or (ite==2*count_threshold):
            break
    print("count",count)
    if len(data["area"])!=0:        
        file = gzip.GzipFile(f'{path}/{data_name}', 'wb')
        pickle.dump(data, file, -1)
        file.close()


