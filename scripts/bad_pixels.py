import os
import numpy as np
import random

if __name__ == "__main__":
    pvalue_range = range(1024)
    feature_dir = '../data/medium/feature/test'
    npy_sample = np.load(os.path.join(feature_dir, os.listdir(feature_dir)[0]))
    H, W = npy_sample.shape
    range_size = H * W
    sel_range = list(range(range_size))
    del sel_range[int(range_size / 2)]

    for corrupt_num in range(10):
        corrupt_dir = os.path.join(feature_dir.split('test')[0], f'corrupt/corrupt_{corrupt_num + 1}')
        if not os.path.exists(corrupt_dir):
            os.makedirs(corrupt_dir)
        
        for i, npy in enumerate(os.listdir(feature_dir)):
            feature = np.load(os.path.join(feature_dir, npy)).reshape(H * W)
            corrupt_pixels = random.sample(sel_range, corrupt_num + 1)
            for cp in corrupt_pixels:
                feature[cp] = random.choice(pvalue_range)
            feature.resize(H, W)
            np.save(os.path.join(corrupt_dir, npy), feature)
        
    