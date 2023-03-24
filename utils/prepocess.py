import numpy as np

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