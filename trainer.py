from __future__ import absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
import numpy as np
from torch import nn
import torch as t
from utils import array_tool as at
# from utils.vis_tool import Visualizer
from model.utils.cutmix import cutmix_generate
from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter
import matplotlib.pyplot as plt
from utils.visual import res_visual
from model.utils.center_tool import generate_map



LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',                        
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
#         self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

        
    def forward(self, imgs, scale, paste_scale, copy_img,  copy_bboxes, copy_labels, paste_img, paste_bboxes, paste_labels, cutmix_flag=False, plot_flag=False, drop_flag=False, crop_flag=False, keep_flag=False, device='cuda'):
          
        

        _, _, H, W = imgs.shape
        img_size = (H, W)        

        features = self.faster_rcnn.extractor(imgs)
        
        features, attentive_maps, attention_maps = self.faster_rcnn.aug(features,img_size)
        
        info = {}
        
        if cutmix_flag:            
            new_imgs, new_bboxes, new_labels, new_scale, info = cutmix_generate(copy_img, scale, paste_scale, paste_img, attention_maps, copy_bboxes, copy_labels, paste_bboxes, paste_labels, info)
                  
            
            if new_imgs is None:
                
                bboxes = copy_bboxes
                labels = copy_labels   
            else:              
                
                imgs = new_imgs                               
                bboxes = new_bboxes
                labels = new_labels
                scale = new_scale                
                _, _, H, W = imgs.shape
                img_size = (H, W)
                center = generate_map(bboxes, img_size)     
                imgs = imgs.to(device)
                bboxes = bboxes.to(device)
                cutmix_features = self.faster_rcnn.extractor(imgs)

                cutmix_features, attentive_maps, attention_maps = self.faster_rcnn.aug(cutmix_features,img_size)
        else:
            info["use_cutmix"] = 0
            bboxes = copy_bboxes
            labels = copy_labels        
        
        center = generate_map(bboxes, list(imgs.shape[2:]))  
#         if cutmix_flag:
#             features = cutmix_features
        
        with t.no_grad():
            if crop_flag:
                crop_image = self.faster_rcnn.batch_aug(imgs, attentive_maps[:, :1, :, ], 'crop', theta=(0.4, 0.6), padding_ratio=0.1)
            else:
                crop_image = None
            if drop_flag:
                drop_image = self.faster_rcnn.batch_aug(imgs, attentive_maps[:, 1:, :, ], 'drop', theta=(0.2, 0.5))
            else:
                drop_image = None
        
        feature_map = []

           
        if not cutmix_flag or keep_flag:
            if cutmix_flag:
                features = t.nn.functional.interpolate(features, cutmix_features.shape[2:])           
            feature_map.append(features)                                    
        if cutmix_flag:            
            feature_map.append(cutmix_features)
        if crop_flag:
            crop_features = self.faster_rcnn.extractor(crop_image)
            feature_map.append(crop_features)
        if drop_flag:
            drop_features = self.faster_rcnn.extractor(drop_image)
            feature_map.append(drop_features)
        
        features2 = t.stack(feature_map, dim=0)        
        rpn_locs = list(range(len(feature_map)))
        rpn_scores = list(range(len(feature_map)))
        rois = list(range(len(feature_map)))
        roi_indices = list(range(len(feature_map)))
        anchors = list(range(len(feature_map)))
        rpn_loc_loss_s = 0
        rpn_cls_loss_s = 0
        roi_loc_loss_s = 0
        roi_cls_loss_s = 0
        center_loss_s = 0
        center_loss_f = nn.MSELoss()
        for i in range(len(feature_map)):
            rpn_locs[i], rpn_scores[i], rois[i], roi_indices[i], anchors[i] = \
            self.faster_rcnn.rpn(features2[i], img_size, scale)

            rpn_locs[i], rpn_scores[i], rois[i], roi_indices[i], anchors[i] = \
                self.faster_rcnn.rpn(features2[i], img_size, scale)

            rpn_locs[i], rpn_scores[i], rois[i], roi_indices[i], anchors[i] = \
                self.faster_rcnn.rpn(features2[i], img_size, scale)

        rpn_locs = t.cat(rpn_locs, dim=0)
        rpn_scores = t.cat(rpn_scores, dim=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        # 仅当保留原feature map，cutmix有效时，有不同的bbox, labels 
        if keep_flag and cutmix_flag and info["use_cutmix"] == 1:
            t1 = [t.from_numpy(paste_bboxes).to(device)]    
            for i in range(len(feature_map)-1):
                t1.append(bboxes.squeeze(dim=0))   
                
            bboxs = t1            
            
            t2 = [t.from_numpy(paste_labels).to(device).reshape([len(paste_bboxes), ])]         
            
            for i in range(len(feature_map)-1):                
                t2.append(labels)                
            labels = t2
        else:
            if len(bboxes)!=3:
                bboxes = bboxes.unsqueeze(dim=0)
            bboxs = t.stack([bboxes]*len(feature_map), dim=0)            
            labels = t.stack([labels]*len(feature_map), dim=0)
        
        for i in range(len(feature_map)):
            bbox = bboxs[i].reshape(-1, 4)                                   
            label = labels[i].reshape(-1, )
            rpn_score = rpn_scores[i]
            rpn_loc = rpn_locs[i]
            roi = rois[i]
            features = features2[i]
            anchor = anchors[i]

            # Sample RoIs and forward
            # it's fine to break the computation graph of rois,
            # consider them as constant input
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi,
                at.tonumpy(bbox),
                at.tonumpy(label),
                self.loc_normalize_mean,
                self.loc_normalize_std)
            # NOTE it's all zero because now it only support for batch=1 now
            sample_roi_index = t.zeros(len(sample_roi))
            
            roi_cls_loc, roi_score = self.faster_rcnn.head(
                features,
                sample_roi,
                sample_roi_index.cuda())

            # ------------------ RPN losses -------------------#
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(bbox),
                anchor,
                img_size)
            gt_rpn_label = at.totensor(gt_rpn_label).long()
            gt_rpn_loc = at.totensor(gt_rpn_loc)
            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma)

            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.to(device), ignore_index=-1)
            _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
            _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
            self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

            # ------------------ ROI losses (fast rcnn loss) -------------------#
            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[t.arange(0, n_sample).long().to(device), \
                                  at.totensor(gt_roi_label).long()]
            gt_roi_label = at.totensor(gt_roi_label).long()
            gt_roi_loc = at.totensor(gt_roi_loc)

            roi_loc_loss = _fast_rcnn_loc_loss(
                roi_loc.contiguous(),
                gt_roi_loc,
                gt_roi_label.data,
                self.roi_sigma)

            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(device))


            self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

            rpn_cls_loss_s += rpn_cls_loss / len(feature_map)
            rpn_loc_loss_s += rpn_loc_loss / len(feature_map)
            roi_loc_loss_s += roi_loc_loss / len(feature_map)
            roi_cls_loss_s += roi_cls_loss / len(feature_map)
            
        attention_map = t.tanh(attention_maps)
 #       center_loss = center_loss_f(attention_map, center).float()
#        losses = [rpn_loc_loss_s, rpn_cls_loss_s, roi_loc_loss_s, roi_cls_loss_s, center_loss]
        losses = [rpn_loc_loss_s, rpn_cls_loss_s, roi_loc_loss_s, roi_cls_loss_s]
        losses = losses + [sum(losses)]
#         print("-"*100)
#         for i in losses:
#             print(i)
#         print("-"*100)
            
        if plot_flag:
            
            res_visual(copy_img, copy_bboxes, copy_labels, paste_img, paste_bboxes, paste_labels, crop_image, drop_image, attention_map, center, imgs, bboxes, labels, features, True, self.faster_rcnn)
        
        return LossTuple(*losses), info, imgs, bboxes, labels
    

    def train_step(self, *args, **kwargs):
        self.optimizer.zero_grad()                
        losses, info, imgs, bboxes, labels = self.forward(*args, **kwargs)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses, info, imgs, bboxes, labels
    
    
    def cutmix_process(self, imgs, scale, paste_scale, copy_img,  copy_bboxes, copy_labels, paste_img, paste_bboxes, paste_labels, threshold,overlap_threshold, device='cuda', crop=True, drop=True):
        features = self.faster_rcnn.extractor(imgs)        
        _, _, H, W = imgs.shape
        img_size = (H, W)
        features, attentive_maps, attention_maps = self.faster_rcnn.aug(features,img_size)        
        info = {}
        new_imgs, new_bboxes, new_labels, _, info = cutmix_generate(copy_img, scale, paste_scale, paste_img, attention_maps, copy_bboxes, copy_labels,paste_bboxes, paste_labels, info, device=device, threshold=threshold, overlap_threshold=overlap_threshold)        
        if crop:
            crop_image = self.faster_rcnn.batch_aug(imgs, attentive_maps[:, :1, :, ], 'crop', theta=(0.4, 0.6), padding_ratio=0.1)
            info["crop_image"]=crop_image
        if drop:
            drop_image = self.faster_rcnn.batch_aug(imgs, attentive_maps[:, 1:, :, ], 'drop', theta=(0.2, 0.5))
            info["drop_image"]=drop_image
        return new_imgs, new_bboxes, new_labels, _, info, attention_maps
        
        

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
#         save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
#         self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, device='cuda'):
        state_dict = t.load(path, map_location=t.device(device))
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
    
    def update_lr(self, lr):
        self.optimizer = self.faster_rcnn.update_lr(lr)
#         return self
        
    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma, device='cuda'):
    in_weight = t.zeros(gt_loc.shape).to(device)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).to(device)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)
