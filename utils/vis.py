import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import data.voc_dataset as dv
import numpy as np
from data.dataset import Dataset, TestDataset, inverse_normalize
from utils import array_tool as at
import torch

def res_visual(copy_image=None, copy_bboxes=None, copy_labels=None, paste_image=None, paste_bboxes=None, paste_labels=None, crop_image=None, drop_image=None, attention_map=None, center=None, input_image=None, input_bboxes=None, input_labels=None, features=None, predict=True, model=None):
    
    if copy_image is not None:
        copy_image = inverse_normalize(at.tonumpy(copy_image.squeeze())) / 255
    if crop_image is not None:        
        crop_image = inverse_normalize(at.tonumpy(crop_image.squeeze())) / 255
    if drop_image is not None:
        drop_image = inverse_normalize(at.tonumpy(drop_image.squeeze())) / 255
    if paste_image is not None:
        paste_image = inverse_normalize(at.tonumpy(paste_image.squeeze())) / 255
    if input_image is not None:
        input_image = inverse_normalize(at.tonumpy(input_image.squeeze())) / 255
    
    
    fig, ax=plt.subplots(2, 4,figsize=(25, 25))
    
    if copy_image is not None:
        ax[0][0].imshow(copy_image.transpose(1, 2, 0))     
        if not isinstance(copy_bboxes, np.ndarray) and not isinstance(copy_bboxes, torch.Tensor):
            copy_bboxes = np.array(copy_bboxes)
        copy_bboxes = copy_bboxes.reshape([-1, 4])
        copy_labels = copy_labels.reshape(-1 , )
        
        w = copy_bboxes[:, 3] - copy_bboxes[:, 1]
        h = copy_bboxes[:, 2] - copy_bboxes[:, 0]
        for i in range(copy_bboxes.shape[0]):                        
            ax[0][0].add_patch(Rectangle(copy_bboxes[i][[1, 0]],w[i], h[i], fill=False,edgecolor='r'))
            
            ax[0][0].text(copy_bboxes[i][1], copy_bboxes[i][0], dv.VOC_BBOX_LABEL_NAMES[copy_labels[i]])
        ax[0][0].set_title("src image")

        ax[0][0].axis("off")

    if crop_image is not None:
        ax[0][1].imshow(crop_image.transpose(1, 2, 0))
        ax[0][1].set_title("crop image")
        ax[0][1].axis("off")
    
    if drop_image is not None:
        ax[0][2].imshow(drop_image.transpose(1, 2, 0))
        ax[0][2].set_title("drop image")
        ax[0][2].axis("off")
    
    if center is not None:
        ax[0][3].imshow(center.detach().cpu().numpy().astype(np.float64), cmap='gray')
        ax[0][3].set_title("center mask")
        ax[0][3].axis("off")
    
    if attention_map is not None:
        ax[1][0].imshow(attention_map.detach().cpu().numpy(), cmap='hot')
        ax[1][0].set_title("attention map image")
        ax[1][0].axis("off")        


    if paste_image is not None:
        ax[1][1].imshow(paste_image.transpose(1, 2, 0))

        w_p = paste_bboxes[:, 3] - paste_bboxes[:, 1]
        h_p = paste_bboxes[:, 2] - paste_bboxes[:, 0]

        for i in range(paste_bboxes.shape[0]):                                                             

            ax[1][1].add_patch(Rectangle(paste_bboxes[i][[1, 0]],w_p[i], h_p[i], fill=False,edgecolor='r'))
            ax[1][1].text(paste_bboxes[i][1], paste_bboxes[i][0], dv.VOC_BBOX_LABEL_NAMES[paste_labels[i]])
        ax[1][1].set_title("paste image")
        ax[1][1].axis("off")        
    
    if input_image is not None:
        
        ax[1][2].imshow(input_image.transpose(1, 2, 0))    
        if not isinstance(input_bboxes, np.ndarray) and not isinstance(input_bboxes, torch.Tensor):
            input_bboxes = np.array(input_bboxes)        
        input_bboxes = input_bboxes.reshape(-1, 4)
        w = input_bboxes[:, 3] - input_bboxes[:, 1]
        h = input_bboxes[:, 2] - input_bboxes[:, 0]    
        for i in range(input_bboxes.shape[0]):                                                             
            ax[1][2].add_patch(Rectangle(input_bboxes[i][[1, 0]],w[i], h[i], fill=False,edgecolor='r'))              
            ax[1][2].text(input_bboxes[i][1], input_bboxes[i][0], dv.VOC_BBOX_LABEL_NAMES[input_labels.reshape(-1, len(input_bboxes))[0][i]])
        ax[1][2].set_title("res image")
        ax[1][2].axis("off")        

    if predict:
        
        sizes = input_image.shape[1:]    
        in_image = torch.from_numpy(input_image).unsqueeze(dim=0).cuda()
        pred_bboxes, pred_labels, pred_scores = model.predict(in_image, [sizes])
        
        ax[1][3].imshow(input_image.transpose(1, 2, 0))                          
        pred_bboxes = np.array(pred_bboxes).reshape(-1, 4)
        w = pred_bboxes[:, 3] - pred_bboxes[:, 1]
        h = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        if pred_bboxes.shape[0] >0:
            for i in range(pred_bboxes.shape[0]):                                 
                ax[1][3].add_patch(Rectangle(pred_bboxes[i][[1, 0]],w[i], h[i], fill=False,edgecolor='r'))
                ax[1][3].text(pred_bboxes[i][1], pred_bboxes[i][0], dv.VOC_BBOX_LABEL_NAMES[pred_labels[0][i]])
        ax[1][3].set_title(f"res image pred")

        ax[1][3].axis("off") 
    else:
        if features is not None:
            _, _, h, w = features.shape
            features_img = torch.zeros([h*5, w*5])
            for i in range(1, 6):
                for j in range(1, 6):
                    features_img[(i-1)*h:h*i, (j-1)*w:w*j] = features[0,(i-1)*5+j-1]
            ax[1][3].imshow(features_img.detach().cpu().numpy())
            ax[1][3].axis("off")
    plt.show()