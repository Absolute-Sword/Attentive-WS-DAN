from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool
import numpy as np
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt
import torch.nn.functional as F
import random


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)  # not 是干嘛的...

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:  # ??
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicDeConv2d(nn.Module):
    def __init__(self):
        super(BasicDeConv2d, self).__init__()
        # self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, kernel_size=1, step=1)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x, image_size):
        x = nn.functional.interpolate(x, image_size)
        return x


class attentive_aug(nn.Module):
    def __init__(self):
        super(attentive_aug, self).__init__()
        # attenion maps = 32, vgg feature maps is 512
        self.attentions = BasicConv2d(512, 32, kernel_size=1)
        self.deconv = BasicDeConv2d()

    def forward(self, feature_map, image_size):
        
        attention_map = self.attentions(feature_map)
        attention_map = self.deconv(attention_map, image_size)        
        attentive_map, attention_map = self.attentive_map_generate(attention_map)
        return feature_map, attentive_map, attention_map.squeeze()

    def attentive_map_generate(self, attention_maps):
        attentive_map = []
        for i in range(attention_maps.shape[0]):
            
            attentive_weights = t.sqrt(attention_maps[i].sum(
                dim=(1, 2)).detach() + 1e-5)  # shape: [C,]
            
            attentive_weights = F.normalize(attentive_weights, p=1, dim=0)
            
            k_index = np.random.choice(
                32, 2, p=attentive_weights.cpu().numpy())  # change random select, change torch random
            attentive_map.append(attention_maps[i, k_index, ...])
        attention_maps = attention_maps.mean(dim=1)
        attentive_map = t.stack(attentive_map, dim=0)

        return attentive_map, attention_maps



class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):

        extractor, classifier = decom_vgg16()
        aug = attentive_aug()
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier)

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            aug,
            rpn,
            head,
        )

    def batch_aug(self, images, attention_maps, mode='crop', theta=0.5, padding_ratio=0.1):
        N, C, H, W = images.shape
        if mode == 'crop':
            crop_images = []
            for i in range(N):
                attention_map = attention_maps[i:i + 1]
                if isinstance(theta, tuple) or isinstance(theta, list):
                    threshold = random.uniform(*theta) * attention_map.max()
                else:
                    threshold = theta * attention_map.max()
                crop_mask = F.upsample_bilinear(attention_map, size=(H, W)) >= threshold
                #  under code is strange
                nonzero_indices = t.nonzero(crop_mask[0, 0, ...])
                height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * H), 0)
                height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * H), H)
                width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * W), 0)
                width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * W), W)
                crop_images.append(
                    F.upsample_bilinear(images[i:i + 1, :, height_min:height_max, width_min:width_max],
                                        size=(H, W)))
            crop_images = t.cat(crop_images, dim=0)
            return crop_images
        elif mode == 'drop':
            drop_masks = []
            for batch_index in range(N):
                atten_map = attention_maps[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_d = random.uniform(*theta) * atten_map.max()
                else:
                    theta_d = theta * atten_map.max()

                drop_masks.append(F.upsample_bilinear(atten_map, size=(H, W)) < theta_d)
            drop_masks = t.cat(drop_masks, dim=0)
            drop_images = images * drop_masks.float()  # float ??
            return drop_images
        else:
            raise Exception("use undefined mode")


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
#         x = x.cuda()
#         rois = rois.cuda()
#         roi_indices = roi_indices.c        
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
            mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
