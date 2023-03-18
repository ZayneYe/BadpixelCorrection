import os
import numpy as np
import torch


class SamsungDataset(torch.utils.data.Dataset):
    def __init__(self, dir, type):
        self.feature_path = os.path.join(dir, 'feature', type)
        # self.label_path = os.path.join(dir, 'label', type)

    def __getitem__(self, index):
        file = os.listdir(self.feature_path)[index]
        feature = np.load(os.path.join(self.feature_path, file))
        # label = np.load(os.path.join(self.label_path, file))
        return feature

    def __len__(self):
        return len(os.listdir(self.feature_path))