import os
import numpy as np
import random

if __name__ == "__main__":
    feature_dir = '../data/medium/feature/test'
    npy_sample = np.load(os.path.join(feature_dir, os.listdir(feature_dir)[0]))

    for corrupt_num in range(10):
        
        corrupt_dir = os.path.join(feature_dir.split('test')[0], f'corrupt/corrupt_{corrupt_num + 1}')
        
        for npy in os.listdir(corrupt_dir):
            feature = np.load(os.path.join(corrupt_dir, npy))
            print(np.sum(feature != npy_sample))
            # print(corrupt_num)
            break
            
        
    