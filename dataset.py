import os
import numpy as np
import torch
from utils.prepocess import *


class SamsungDatasetSmall(torch.utils.data.Dataset):
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

class SamsungDataset15x15(torch.utils.data.Dataset):
    def __init__(self, dir, type, cluster_size=5, transform=None):
        self.feature_path = os.path.join(dir, type)
        # self.label_path = os.path.join(dir, 'label', type)
        self.cluster_size = cluster_size
        self.transform = transform

    def __getitem__(self, index):
        file = os.listdir(self.feature_path)[index]
        feature = np.load(os.path.join(self.feature_path, file))
        H, W = feature.shape
        start_defect_h = H//2 - self.cluster_size // 2
        start_defect_w = W//2 - self.cluster_size // 2
        label = feature[start_defect_h:start_defect_h + self.cluster_size, start_defect_w:start_defect_w + self.cluster_size]
        if self.transform:
            feature = self.transform(feature)
        return feature, label
            
    def __len__(self):
        return len(os.listdir(self.feature_path))
    

def divide64(fea, idx):
    patch = 8
    fea_combine = []
    H, W = fea.shape
    H_slice = int(H / patch)
    W_slice = int(W / patch)
    for i in range(patch):
        row_start = i * H_slice
        row_end = (i + 1) * H_slice
        
        for j in range(patch):
            col_start = j * W_slice
            col_end = (j + 1) * W_slice
            fea_sliced = fea[row_start:row_end, col_start:col_end]
            fea_combine.append(fea_sliced)
    return fea_combine[idx]
            
def divide4(fea, idx):
    H, W = fea.shape
    fea1 = fea[0:int(H/2 + 2), 0:int(W/2 + 2)]
    fea2 = fea[int(H/2 - 2):H, 0:int(W/2 + 2)]
    fea3 = fea[0:int(H/2 + 2), int(W/2 - 2):W]
    fea4 = fea[int(H/2 - 2):H, int(W/2 - 2):W]
    fea_combine = np.stack((fea1, fea2, fea3, fea4))
    return fea_combine[idx]

def preprocess(org_img, img, mask, idx, patch_num=64):
    # org_img, img, mask = resize(org_img, img, mask)
    if patch_num == 4:
        org_img_ = divide4(org_img, idx)
        img_ = divide4(img, idx)
        mask_ = divide4(mask, idx)
    elif patch_num == 64:
        org_img_ = divide64(org_img, idx)
        img_ = divide64(img, idx)
        mask_ = divide64(mask, idx)
    return org_img_, img_, mask_


def get_original_imgs_path(dataset):
    if dataset == 'S7-ISP':
        return '/data1/Bad_Pixel_Detection/ISP_0.7/original_imgs'
    elif dataset == 'FiveK':
        return '/data1/Bad_Pixel_Detection/data/FiveK/original_images'
    else:
        assert False, 'dataset not found!'

class SamsungDataset(torch.utils.data.Dataset):
    def __init__(self, dir, cate, transform=None, mask_transform=None, patch_num=4, dataset='S7-ISP'):
        self.transform = transform
        self.mask_transform = mask_transform
        self.patch_num = patch_num
        # self.original_imgs = '/data1/Bad_Pixel_Detection/data/ISP_0.7/original_imgs'
        # self.original_imgs = '/data1/Bad_Pixel_Detection/data/FiveK/original_images'
        self.original_imgs = get_original_imgs_path(dataset)
        self.imgs_path = os.path.join(dir, 'imgs', cate)
        self.masks_path = os.path.join(dir, 'masks', cate)
        

    def __getitem__(self, index):
        idx = index % self.patch_num
        file = os.listdir(self.imgs_path)[index // self.patch_num]
        # mask_file = os.listdir(self.masks_path)[index // self.patch_num]
        img = np.load(os.path.join(self.imgs_path, file)).astype(np.float32)
        mask = np.load(os.path.join(self.masks_path, file)).astype(np.float32)
        org_img = np.load(os.path.join(self.original_imgs, file)).astype(np.float32)
        if self.patch_num != 1:
            org_img, img, mask = preprocess(org_img, img, mask, idx, self.patch_num)
        
        if self.transform:
            org_img = self.transform(org_img)
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return org_img, img, mask, file

    def __len__(self):
        return self.patch_num * len(os.listdir(self.imgs_path))