import rawpy
import os
import numpy as np
import math
import random
import sys

def compute_pixel(data_dir):
    max_pixel, min_pixel = 0, sys.maxsize
    for dng in os.listdir(data_dir):
        raw = rawpy.imread(os.path.join(data_dir, dng))
        raw_data = raw.raw_image
        if np.max(raw_data) > max_pixel:
            max_pixel = np.max(raw_data)
        if np.min(raw_data) < min_pixel:
            min_pixel = np.min(raw_data)
        return max_pixel, min_pixel

def resize_dng(raw_data, cut_size):
    # print(raw_data.shape)
    H, W = raw_data.shape
    H_, W_ = H, W
    while H_ % cut_size:
        H_ += 1
    while W_ % cut_size:
        W_ += 1
    dng_resized = np.zeros((H_, W_))
    h_start = math.floor((H_ - H) / 2)
    w_start = math.floor((W_ - W) / 2)
    
    dng_resized[h_start:H + h_start, w_start:W+w_start] = raw_data
    return dng_resized


if __name__ == "__main__":
    random.seed(77)
    cut_size = 15
    sample_amt = 125 #should be a square number

    data_dir = '/data1/FiveK/Canon_EOS_5D/DNG'
    feature_dir = f'/data1/Bad_Pixel_Correction/FiveK/feature_{cut_size}'
   
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    split_dng = np.zeros((cut_size, cut_size))
    maxv, minv = compute_pixel(data_dir)
    cnt = 0
    for dng in os.listdir(data_dir):
        raw = rawpy.imread(os.path.join(data_dir, dng))
        raw_data = raw.raw_image
        resized_data = resize_dng(raw_data, cut_size)
        H, W = resized_data.shape
        cut_data = np.zeros((cut_size, cut_size))
        
        i_sel = random.sample(range(0, H - cut_size, cut_size), int(pow(sample_amt, 0.5)))
        j_sel = random.sample(range(0, W - cut_size, cut_size), int(pow(sample_amt, 0.5)))
        
        for i in i_sel:
            for j in j_sel:
                npy = f"{dng.split('.')[0]}_{i}_{j}.npy"
                cut_data = resized_data[i:i+cut_size, j:j+cut_size]
                # bad_data = np.copy(cut_data)
                # bad_data[int(cut_size / 2)][int(cut_size / 2)] = random.choice((maxv, minv))
                # print(cut_data.all() == bad_data.all()) 
                np.save(os.path.join(feature_dir, npy), cut_data)
                cnt += 1
                print(f"{cnt} npy file has been created.")
                


        
        