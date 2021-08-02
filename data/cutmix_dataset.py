import os
import torch
from data import util
from torch.utils.data import Dataset
import numpy as np
from utils.config import opt
from tqdm import tqdm
import pickle
from pickle import UnpicklingError
import gzip


class Cutmix_dataset(Dataset):
    def __init__(self, path="cutmix_data", scale=None, files=None, length=-1, gzip_flag=False):

        file_names = os.listdir(path)

        self.gzip_flag = gzip_flag
        self.path = path
        self.length = length
        self.scale = scale

        if files is not None:
            self.file_names = files
        else:
            if scale is not None:
                self.file_names = file_names[:int(len(file_names) * scale)]

    def __getitem__(self, index):
        images = []
        bboxes = []
        labels = []
        scales = []
        file_name = self.file_names[index]
        if self.gzip_flag:
            file = gzip.GzipFile(os.path.join(self.path, file_name), "rb")
        else:
            file = open(os.path.join(self.path, file_name), "rb")
        try:
            data = pickle.load(file)
            images.extend(data["imgs"][:self.length])
            bboxes.extend(data["bboxes"][:self.length])
            labels.extend(data["labels"][:self.length])
            scales.extend(data["scales"][:self.length])
        except Exception as e:
            print(f"load {file_name} {e}")
        file.close()
        return images, bboxes, labels, scales

    def __len__(self):
        return len(self.file_names)
