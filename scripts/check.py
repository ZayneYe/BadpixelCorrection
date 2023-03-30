import os
import numpy as np
import random

if __name__ == "__main__":
    mode = 'posion'
    feature_dir = '../data/medium2/feature_13/train'
    if mode == 'corrupt':
        npy_sample = np.load(os.path.join(feature_dir, os.listdir(feature_dir)[0]))

        for corrupt_num in range(10):
            
            corrupt_dir = os.path.join(feature_dir.split('test')[0], f'corrupt/corrupt_{corrupt_num + 1}')
            
            for npy in os.listdir(corrupt_dir):
                feature = np.load(os.path.join(corrupt_dir, npy))
                print(np.sum(feature != npy_sample))
                # print(corrupt_num)
                break
    
    elif mode == 'posion':
        cate = 'train'
        corrupt_dir = os.path.join(feature_dir.split(cate)[0], f'poison_{cate}')
        correct_vec = {0:0, 1:0, 2:0, 'wrong':0}
        for npy in os.listdir(feature_dir):
            npy_sample = np.load(os.path.join(feature_dir, npy))
            feature = np.load(os.path.join(corrupt_dir, npy))
            if np.sum(feature != npy_sample) == 0:
                correct_vec[0] += 1
            elif np.sum(feature != npy_sample) == 1:
                correct_vec[1] += 1
            elif np.sum(feature != npy_sample) == 2:
                correct_vec[2] += 1
            else:
                correct_vec['wrong'] += 1
        print(correct_vec)
    
    else:
        print("No such mode.")    
    