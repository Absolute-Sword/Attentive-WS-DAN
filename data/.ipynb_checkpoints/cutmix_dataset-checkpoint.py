import os
import torch
from data import util
from torch.utils.data import Dataset
import numpy as np
from utils.config import opt
from tqdm.notebook import tqdm
import pickle
from pickle import UnpicklingError

class Cutmix_dataset(Dataset):
    def __init__(self, scale=None, files=None, length=-1):
        
        file_names = os.listdir("cutmix_data")
        self.images = []
        self.bboxes = []
        self.labels = []
        self.scales = []
        
        if files is not None:
            file_names = files
        else:
            if scale is not None:
                file_names = file_names[:scale]
        for file_name in tqdm(file_names):
            print(file_name)
            file = open(os.path.join("cutmix_data",file_name), "rb")
            try:
                data = pickle.load(file)
                self.images.extend(data["imgs"][:length])
                self.bboxes.extend(data["bboxes"][:length])
                self.labels.extend(data["labels"][:length])
                self.scales.extend(data["scales"][:length])
            except Exception as e:
                print(f"load {file_name} {e}")
            file.close()
                    
            
    def __getitem__(self, index):
        return self.images[index], self.bboxes[index], self.labels[index], self.scales[index]
    def __len__(self):
        return len(self.images)