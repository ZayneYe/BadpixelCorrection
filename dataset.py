import os
import numpy as np
import torch
from utils.prepocess import *


class SamsungDataset(torch.utils.data.Dataset):
    def __init__(self, dir, type):
        self.feature_path = os.path.join(dir, type)
        # self.label_path = os.path.join(dir, 'label', type)

    def __getitem__(self, index):
        file = os.listdir(self.feature_path)[index]
        feature = np.load(os.path.join(self.feature_path, file))
        feature, label = prepocess(feature)
        return feature, label

    def __len__(self):
        return len(os.listdir(self.feature_path))