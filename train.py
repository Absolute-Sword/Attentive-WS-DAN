from __future__ import  absolute_import
import os
from model.utils.center_tool import generate_map
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
# from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import torch
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
import logging
import time
import warnings
import numpy as np

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
run_name = 'train for backbone 0.8 data'
# matplotlib.use('TkAgg')
logging.basicConfig(
        filename=os.path.join("./log", f"{run_name}_{time.strftime('%Y-%m-%d-%H-%M')}-log"),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
warnings.filterwarnings("ignore")
    
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


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print(f'load data, data:length {len(dataset)}')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    print(f'load test_data, test_data:length {len(testset)}')
    test_dataloader = data_.DataLoader(testset,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False, \
                                        # pin_memory=True
                                        )    
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    best_map = 0
    lr_ = opt.lr
    
    plot_flag = False
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        index = list(range(len(dataset))) 
        np.random.shuffle(index)
        loss_history = []
        count = 0
        
#         if epoch > 10:
#             cutmix_flag =True
        flag = torch.randn(len(dataloader))
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
#             if (ii + 1) % plot_every == 0:  
#                 plot_flag = True
#             else:
            
                
#             cutmix_flag = True if flag[ii] > 0.4 else False
            cutmix_flag = False

            scale = at.scalar(scale)
            img, bbox, labels = img.cuda().float(), bbox_.cuda(), label_.cuda()
            
                                
            paste_img, paste_bboxes, paste_labels, paste_difficult = dataset.db.get_example(index[ii])
            paste_img, paste_bboxes, paste_labels, paste_scale = dataset.tsf((paste_img, paste_bboxes, paste_labels))            
            copy_cache = [img, bbox, labels]
            paste_cache = [paste_img, paste_bboxes, paste_labels]
            losses, info,*_ = trainer.train_step(img, scale, paste_scale, *copy_cache, *paste_cache, cutmix_flag, plot_flag)
            
            if info["use_cutmix"] == 1:
                count += 1
#                 count_x.append(ii)
#                 count_y.append(losses.total_loss.item())
                
            loss_history.append(losses.total_loss.item())
        
            if (ii + 1) % opt.plot_every == 0:
                
                logging.info(f"[Batch: {epoch}/Iter {ii + 1}] training loss: {np.mean(loss_history):.2f} cutmix account {count}")   
                                


        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        logging.info(f"[Batch: {epoch}] eval loss: {eval_result['map']:.4f}")# 注意这里是会进行四舍五入
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        count = 0

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            if best_map < 0.4:
                best_path = None
            else:
                best_path = trainer.save(best_map=best_map)
#        if epoch == 18:
#            if best_path is not None:
#                trainer.load(best_path)
#            trainer.faster_rcnn.scale_lr(opt.lr_decay)
#            lr_ = lr_ * opt.lr_decay



if __name__ == '__main__':

    
    train()
