import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data.dataset import inverse_normalize
from utils import array_tool as at
from model.utils.bbox_tools import bbox_iou
from skimage import transform as sktsf
 
def pytorch_normalze(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(img)
    return img
 
def cutmix_generate(img, scale, paste_scale, paste_img, attention_map, bboxes, labels, paste_bboxes, paste_labels, info, bbox_drop=True, rescale=True, device='cuda', threshold=0.7, overlap_threshold=0.7):
    src_img = inverse_normalize(at.tonumpy(img.squeeze()))    
    src_img = torch.from_numpy(src_img).to(device)
    target_img = inverse_normalize(at.tonumpy(paste_img.squeeze()))
    target_img = torch.from_numpy(target_img.squeeze().copy()).to(device)
    target_img_copy = target_img.clone()
     
    src_size = src_img.shape    
    target_size = target_img.shape    
    img_size = [target_size[1] if target_size[1] > src_size[1] else src_size[1],
                target_size[2] if target_size[2] > src_size[2] else src_size[2]]
    if target_size[1] >= img_size[1] and target_size[2] >= src_size[2]:
        new_scale = paste_scale
    else:
        new_scale = scale
    
    attention_map = torch.nn.functional.interpolate(attention_map.unsqueeze(0).unsqueeze(0), img_size).squeeze()
    mask = torch.zeros_like(attention_map).to(device)
#     print(attention_map.max(), attention_map.min(), attention_map.mean())
    mean_threshold = attention_map.mean()
    mask[attention_map > mean_threshold] = 1
    mask[attention_map <= mean_threshold] = 0  # mask表示重要的区域    
#     print(torch.sum(mask==1)/mask.shape[0]/mask.shape[1])
    
#     if torch.sum(mask)/mask.shape[0]/mask.shape[1] > 0.9:
#         info["use_cutmix"] = 0
#         res_img = pytorch_normalze(target_img_copy/255).unsqueeze(dim=0)
#         return res_img, torch.from_numpy(paste_bboxes), torch.from_numpy(paste_labels), paste_scale, info
    
    mask = mask.int()
    
    x_scale = img_size[0] / src_size[1]
    y_scale = img_size[1] / src_size[2] 
    
    x_scale_p = img_size[0] / target_size[1]
    y_scale_p = img_size[1] / target_size[2] 
    
    h, w= img_size    
    
    src_img = torch.nn.functional.interpolate(src_img.unsqueeze(0), img_size).squeeze()
    
    target_img = torch.nn.functional.interpolate(target_img.unsqueeze(0), img_size).squeeze()    
    
    
    
    if len(bboxes.shape)==2 and bboxes.shape[1] == 4:
        pass
    else:
        bboxes = bboxes.reshape([-1, 4])
    bboxes = bboxes.int()
    labels = labels.reshape(-1, )
    ori_mask = torch.zeros(src_img.shape[1:]).to(device)
    new_bboxes = []
    new_labels = []
    paste_bboxes_cp = paste_bboxes.copy()
    bboxes_cp = bboxes.clone().detach()
    res_img = target_img.clone()   
    

    b_h = (bboxes[:, 3] - bboxes[:,1]).unsqueeze(dim=1)    
    b_w = (bboxes[:,2] - bboxes[:,0]).unsqueeze(dim=1)
    b = torch.cat([b_h,b_w], axis=1)    
    areas = torch.prod(b,1)
    
    _, index = torch.sort(areas)
    for i in index:                        
        bboxes_cp[i, [0, 2]] = (bboxes[i, [0, 2]] * x_scale).int()
        bboxes_cp[i, [1, 3]] = (bboxes[i, [1, 3]] * y_scale).int()
        ymin, xmin, ymax, xmax = bboxes_cp[i].int()                     
        x0 = ymin.item()
        x1 = ymax.item()
        y0 = xmin.item()
        y1 = xmax.item()        
        ori_mask[x0:x1, y0:y1] = 1    
        ori_mask = ori_mask.int()
        area = (x1-x0)*(y1-y0)
#         print(area/w/h)
        
        
        portion = torch.sum(mask[x0:x1, y0:y1] & ori_mask[x0:x1, y0:y1]) / area

        if bbox_drop:
            remain_flag = True if np.random.rand(1) > 0.2 else False            
        else:
            remain_flag = True
        if rescale:
            rescale_flag = True if np.random.rand(1) > 0.0 else False            
        else:
            rescale_flag = False
        if rescale_flag:
            if np.random.randn(1) >= 0.5:
                    rescale_conf = np.random.randint(low=10, high=1000,size=1)[0]*0.0001 + 1
            else:
                rescale_conf = np.random.rand(1)[0]
#             print(rescale_conf)
            
        if  portion < threshold or not remain_flag or area / h / w > 0.8: # 不考虑占比大的bbox:        
            
            mask[x0:x1, y0:y1] = 0
            for j in range(len(bboxes_cp)):               
                x = np.maximum(x0, bboxes_cp[j][0].detach().cpu().numpy())
                y = np.maximum(y0, bboxes_cp[j][1].detach().cpu().numpy())
                x = np.minimum(x1, bboxes_cp[j][2].detach().cpu().numpy())
                y = np.minimum(y1, bboxes_cp[j][3].detach().cpu().numpy())
                mask[x:x, y:y] = 1
        else:   
            
            index_y = torch.where(mask[ymin:ymax, xmin:xmax]==1)[0].unique().sort()[0] + ymin
            index_x = torch.where(mask[ymin:ymax, xmin:xmax]==1)[1].unique().sort()[0] + xmin # bboxes尽量取得靠近图像边缘
            mask_m = (mask[ymin:ymax, xmin:xmax] & ori_mask[ymin:ymax, xmin:xmax]).bool()

            mask_m = torch.stack([mask_m, mask_m, mask_m], dim=0)
            sub_obj = src_img[:, ymin:ymax, xmin:xmax] # 取出来的目标
            
            if len(index_y)<2 or len(index_x)<2:
                print('!',index_y, type(index_y))
            
            if rescale_flag and 0 < ymin * rescale_conf < h and 0 < ymax * rescale_conf < h and 0 < xmin * rescale_conf < w and 0 < xmax * rescale_conf < w and (ymax-ymin)*rescale_conf*(xmax-xmin)*rescale_conf>400:
                
                index_y = (index_y * rescale_conf).int()
                index_x = (index_x * rescale_conf).int()
                bbox_w = xmax.item() - xmin.item()
                bbox_h = ymax.item() - ymin.item()                
#                 print("before", ymin.item(), ymax.item(), xmin.item(), xmax.item())
                ymin = int(ymin * rescale_conf)
                ymax = int(ymax * rescale_conf)
                xmin = int(xmin * rescale_conf)
                xmax = int(xmax * rescale_conf)    
                
#                 print("enlarge", ymin, ymax, xmin, xmax)
#                 print(bbox_h * rescale_conf,  bbox_w *rescale_conf)
                resize_f = tvtsf.Resize([int(bbox_h*rescale_conf),  int(bbox_w*rescale_conf)])                
                r_mask_m = resize_f(mask_m).bool()
#                 print(r_mask_m.shape, res_img[:, ymin:ymin+int(bbox_h*rescale_conf), xmin:xmin+int(bbox_w*rescale_conf)].shape)
                
                res_img[:, ymin:ymin+int(bbox_h*rescale_conf), xmin:xmin+int(bbox_w*rescale_conf)][r_mask_m] = resize_f(sub_obj)[r_mask_m]
                
            else:
#                 print(ymax*rescale_conf, ymin*rescale_conf, h)
#                 print(xmax*rescale_conf, xmin*rescale_conf, w)
                res_img[:, ymin:ymax, xmin:xmax][mask_m] = sub_obj[mask_m]
            
            new_bboxes.append([index_y[0].item(), index_x[0].item(), index_y[-1].item(), index_x[-1].item()])            
            new_labels.append(labels[i])            
            
            
            
    
    new_bboxes_cp = np.concatenate([new_bboxes])
    mask = mask & ori_mask
    for i in range(paste_bboxes.shape[0]):
        paste_bboxes_cp[i, [0, 2]] = (paste_bboxes[i, [0, 2]] * x_scale_p).astype(np.int)
        paste_bboxes_cp[i, [1, 3]] = (paste_bboxes[i, [1, 3]] * y_scale_p).astype(np.int)
        if len(new_bboxes_cp)>0:
            ymin, xmin, ymax, xmax = paste_bboxes_cp[i]                      
            area = (ymax - ymin) * (xmax - xmin)        
            overlap = torch.sum(mask[int(ymin):int(ymax),int(xmin):int(xmax)] == 1) / area
#             print(overlap)
#             lt = np.maximum(paste_bboxes_cp[i:i+1:,:2], new_bboxes_cp[:,:2])
#             rb = np.minimum(paste_bboxes_cp[i:i+1:,2:], new_bboxes_cp[:,2:])
#             overlap = np.prod(rb - lt, axis=1) / area
            
            if (overlap > overlap_threshold).any():
                continue
#         if np.sum(mask[ymix:ymax, xmax:xmax] == 1) / area > 0.6 :
#             continue                    
        new_bboxes.append(paste_bboxes_cp[i])
        new_labels.append(paste_labels[i])
        
    if len(new_bboxes) == 0:
        new_bboxes = None
    else:
        new_bboxes = torch.from_numpy(np.concatenate([new_bboxes])).float().to(device).unsqueeze(dim=0)
        new_labels = torch.Tensor(new_labels).int().to(device).unsqueeze(dim=0)
        
    mask = mask.int()
    new_mask = mask
#     res_img = src_img.mul(new_mask) + target_img.mul(1 - new_mask)    
    area = torch.sum(new_mask == 1).float().item()/ (torch.sum(ori_mask == 1)+1e-8).float().item()
    
    # compute iou
    res_img = res_img / 255
    res_img = res_img.float()
    
#     if area < 0.6 or len(new_bboxes)==0:   
#     print(area)
    if len(new_bboxes) == 0:  
        info["use_cutmix"] = 0
        res_img = pytorch_normalze(target_img_copy/255).unsqueeze(dim=0)
        return res_img, torch.from_numpy(paste_bboxes), torch.from_numpy(paste_labels), paste_scale, info
        
    else:        
        info["use_cutmix"] = 1
        res_img = pytorch_normalze(res_img).unsqueeze(dim=0)        
        return res_img, new_bboxes, new_labels, new_scale, info