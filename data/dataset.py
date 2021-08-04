from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
import cv2

class Resize(object):
    def   __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        _, h_org , w_org = img.shape

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        img = img.transpose(1, 2, 0)
        image_resized = cv2.resize(img, (resize_w, resize_h))
        image_resized = image_resized.transpose(2, 0, 1)
        
        image_paded = np.full((3, self.h_target, self.w_target), 128.0)
        
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        
        image_paded[:, dh:resize_h + dh, dw:resize_w + dw] = image_resized       
        
        if self.correct_box:
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dw
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dh
            return image_paded, bboxes, resize_ratio
        return image_paded
    
def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


# def preprocess(img, bboxes, target_size):
def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
#     img = sktsf.resize(img, (C, max_size, max_size), mode='reflect',anti_aliasing=False)
#     img, bboxes, scale = Resize([target_size, target_size], True)(np.copy(img), np.copy(bboxes))
    
    img = img / 255.
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
#     def __init__(self, target_size=800):
        self.min_size = min_size
        self.max_size = max_size
#         self.target_size = target_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        

#         img, bbox, scale = preprocess(img, bbox, self.target_size)
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, scale=opt.scale)
        self.tsf = Transform(opt.min_size, opt.max_size)
#         self.tsf = Transform(opt.img_size)

    def __getitem__(self, idx):

        ori_img, ori_bbox, label, difficult = self.db.get_example(idx)
        
        img, bbox, label, scale = self.tsf((ori_img, ori_bbox, label))
        #s TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return int(len(self.db))


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult, scale=opt.test_scale)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img  = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
