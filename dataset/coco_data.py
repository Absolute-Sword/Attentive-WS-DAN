import os
import cv2
from torchvision.datasets import CocoDetection

class CocoDetectionCP(CocoDetection):
    def __init__(self, root_dir, annFile_dir, transforms=None, target_transform=None):
        super(CocoDetectionCP, self).__init__(root_dir, annFile_dir, transforms, target_transform)
        for img_id in self._ids:
            self.coco.getAnnids