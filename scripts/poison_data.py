import os
import numpy as np
import random
import math

def myround(amt):
    if amt - math.floor(amt) >= 0.5:
        return math.ceil(amt)
    else:
        return math.floor(amt)

def bad_pixel(feature_dir, feature_vec, poison_dir, sel_range, bad_num):
    maxv, minv = 0, 1023
    print(len(feature_vec))
    for i, npy in enumerate(feature_vec):
        feature = np.load(os.path.join(feature_dir, npy)).reshape(H * W)
        corrupt_pixels = random.sample(sel_range, bad_num)
        for cp in corrupt_pixels:
            # feature[cp] = random.choice(pvalue_range)  
            feature[cp] = random.choice((minv, maxv))  
        feature.resize(H, W)
        np.save(os.path.join(poison_dir, npy), feature)
        # print(f'{i} posioned .npy file saved.')


if __name__ == "__main__":
    cate = 'val'
    feature_dir = os.path.join('../data/medium2/feature_5', cate)
    poison_dir = os.path.join(feature_dir.split(cate)[0], f'poison_{cate}')
    npy_sample = np.load(os.path.join(feature_dir, os.listdir(feature_dir)[0]))
    H, W = npy_sample.shape
    range_size = H * W
    sel_range = list(range(range_size))
    del sel_range[int(range_size / 2)]

    amt = len(os.listdir(feature_dir))
    amt_0 = myround(amt * 0.99)
    amt_1 = myround(amt * 0.008) 
    print(amt_0, amt_1, amt - amt_0 - amt_1)

    if not os.path.exists(poison_dir):
        os.makedirs(poison_dir)
    
    feature_vec = os.listdir(feature_dir)
    random.seed(66)
    random.shuffle(feature_vec)
    bad_pixel(feature_dir, feature_vec[:amt_0], poison_dir, sel_range, 0)
    bad_pixel(feature_dir, feature_vec[amt_0:amt_0+amt_1], poison_dir, sel_range, 1)
    bad_pixel(feature_dir, feature_vec[amt_0+amt_1:], poison_dir, sel_range, 2)