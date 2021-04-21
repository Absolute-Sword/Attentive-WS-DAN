from torch.utils.data import dataset
from torchvision.datasets import CocoDetection
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

COCO_CLASS_LABELS = ('background',
                     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                     'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                     'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                     'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                     'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                     'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def has_valid_annotation(annos):
    # 一个anno存储多个目标， 每个目标存储'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'个特征
    if len(annos) == 0:
        return False
    # TO-DO: w, h >0 < image.size area > 0
    return ~all(any(s <= 1 for s in anno["bbox"][2:]) for anno in annos)  # 一个目标有一条变小于1即无效，一张图像如果所有物体都为无效则该图像无效


class CocoDetectionCP(CocoDetection):
    def __init__(self, root_dir, annFile, transforms, name="train2017"):
        super(CocoDetectionCP, self).__init__(root_dir, annFile, transforms)
        self.root_dir = root_dir
        self.data_dir = os.path.join(self.root_dir, name)
        valid_ids = []  # self.ids store image id
        for img_id in self.ids:
            anno_ids = self.coco.getAnnIds(img_id)  # one image to more anno
            anno = self.coco.loadAnns(anno_ids)
            if has_valid_annotation(anno):
                valid_ids.append(img_id)
            self.ids = valid_ids

    def __getitem__(self, index):
        # x, y, h, w, center
        image_id = self.ids[index]
        image = self.push_image(image_id)
        annotation = self.push_anno(image_id)
        image = image[:, :, (2, 1, 0)]
        for i in range(len(annotation)):
            x, y, h, w = annotation[i]["bbox"]
            class_id = annotation[i]["category_id"]
            target = []


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def visual(img, target_anno, pred_anno=None):
        # TO_-DO: 随机选取颜色
        for i in range(len(target_anno)):
            box_anno = target_anno[i]["bbox"]
            category = COCO_CLASS_LABELS[target_anno[i]['category_id']]
            text_size = cv2.getTextSize(category, cv2.FONT_ITALIC, 0.55, 1)[0]

            lt = tuple([int(i) for i in box_anno[:2]])
            text_lt = tuple([lt[0], lt[1] - text_size[1]])  # x正方向是往下
            text_rb = tuple([lt[0] + text_size[0], lt[1]])  # y方向是往右
            rb = tuple([int(box_anno[i] + l) for i, l in enumerate(box_anno[2:])])

            cv2.rectangle(img, lt, rb, [255, 255, 0], 1)  # 坐标必须是tuple，int
            cv2.rectangle(img, text_lt, text_rb, [255, 255, 0], cv2.FILLED)
            cv2.putText(img, category, lt, cv2.FONT_ITALIC, 0.5,
                        (255, 255, 255))  # 给定位置字的开始左下角

    def test(self, random=True, item=None, iterable=True):  # item: image_id
        while True:
            if random:
                image_id = np.random.choice(self.ids, 1)[0]
            else:
                image_id = item
            image = self.push_image(image_id)
            assert image is not None
            annotation = self.push_anno(image_id)
            self.visual(image, annotation)
            cv2.imshow(f"{image_id:012}", image)
            if iterable and (cv2.waitKey(0) & 0xFF == ord('q')):
                break
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()

    def push_image(self, item):  # item: image_id
        image = cv2.imread(os.path.join(self.data_dir, f"{item:012}.jpg"))
        return image

    def push_anno(self, item):  # item: image_id
        anno_id = self.coco.getAnnIds(item)
        annotation = self.coco.loadAnns(anno_id)
        return annotation


if __name__ == '__main__':
    ccp = CocoDetectionCP('/home/yuki/Documents/Dataset/coco/',
                          '/home/yuki/Documents/Dataset/coco/annotations/instances_val2017.json', None, name="val2017")
    ccp.test()
