import numpy as np
import torch.nn.functional as F
import torch

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return np.concatenate((arr1,arr2))

def prepocess(feature):
    H, W = feature.shape
    feature = feature.reshape(H*W)
    label = feature[int((H*W - 1) / 2)]
    feature = del_tensor_ele(feature, int((H*W - 1) / 2))
    return feature, label

def resize(org_img, img, mask):
    maxH, maxW = 3024, 4032
    H, W = img.shape
    if H == maxH and W == maxW:
        return org_img, img, mask
    else:
        # if H > maxH or W > maxW:      
        #     org_img_crop = F.crop(torch.tensor(org_img), height=maxH, width=maxW)         
        #     img_crop = F.crop(torch.tensor(img), height=maxH, width=maxW)
        #     mask_crop = F.crop(torch.tensor(mask), height=maxH, width=maxW)
        #     return org_img_crop.numpy(), img_crop.numpy(), mask_crop.numpy()
        # else:
        pad_left = pad_right = (maxW - W) // 2
        pad_top = pad_bottom = (maxH - H) // 2
        if W % 2:
            pad_right += 1
        if H % 2:
            pad_bottom += 1
        org_img = torch.tensor(org_img)
        img = torch.tensor(img)
        mask = torch.tensor(mask)
        org_img_pad = F.pad(org_img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        img_pad = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        mask_pad = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return org_img_pad.numpy(), img_pad.numpy(), mask_pad.numpy()
        
# def prepocess(feature):
#     H, W = feature.shape
#     # feature = feature.reshape(H*W)
#     label = feature[int(H/2), int(W/2)]
#     feature[int(H/2), int(W/2)] = 0
#     # feature = del_tensor_ele(feature, int((H*W - 1) / 2))
#     return feature, label