import os
import numpy as np
import random
import math

def myround(amt):
    if amt - math.floor(amt) >= 0.5:
        return math.ceil(amt)
    else:
        return math.floor(amt)

def decide_range(value, delta, maxv=1023):
    if (1 + delta) * value > maxv:
        return [(0, int((1 - delta) * value))]
    else:
        return [(0, int((1 - delta) * value)), (int((1 + delta) * value), maxv)]
    
def bad_pixel(feature_dir, feature_vec, poison_dir, sel_range, bad_num, delta):
    # maxv, minv = 0, 1023
    maxv, minv = 4095, 0 # FiveK Canon
    
    for i, npy in enumerate(feature_vec):
        feature = np.load(os.path.join(feature_dir, npy)).reshape(H * W)
        corrupt_pixels = random.sample(sel_range, bad_num)
        for cp in corrupt_pixels:
            # feature[cp] = random.choice(pvalue_range)  
            # feature[cp] = random.choice((minv, maxv)) 
            ranges = decide_range(feature[cp], delta, maxv)
            chosed_range = random.choice(ranges)
            b_val = random.randint(*chosed_range) 
            feature[cp] = b_val
        feature.resize(H, W)
        np.save(os.path.join(poison_dir, npy), feature)
        # print(f'{i} posioned .npy file saved.')
    print(f'{len(feature_vec)} corrupted .npy file with {bad_num} bad pixels has generated.')


if __name__ == "__main__":
    use_distribution = False
    error_percentage = 0.85
    total_pixels = 81 # 13*13 patch
    bad_num = int(error_percentage * total_pixels)
    print(bad_num)
    delta=0.7
    cate = 'val'
    feature_dir = os.path.join('/data1/Bad_Pixel_Correction/FiveK/feature_9', cate)
    poison_dir = os.path.join(feature_dir.split(cate)[0], f'poison_{cate}_{error_percentage}')
    npy_sample = np.load(os.path.join(feature_dir, os.listdir(feature_dir)[0]))
    H, W = npy_sample.shape
    range_size = H * W
    sel_range = list(range(range_size))
    del sel_range[int(range_size / 2)]

    amt = len(os.listdir(feature_dir))
    amt_0 = myround(amt * 0.99)
    amt_1 = myround(amt * 0.008)
    if use_distribution:
        print(f'{amt} .npy file in total, splited to: 0 bad pixel: {amt_0} , 1 bad pixel: {amt_1}, 2 bad pixel: {amt - amt_0 - amt_1}')

    if not os.path.exists(poison_dir):
        os.makedirs(poison_dir)
    
    feature_vec = os.listdir(feature_dir)
    random.seed(66)
    random.shuffle(feature_vec)
    if use_distribution:
        bad_pixel(feature_dir, feature_vec[:amt_0], poison_dir, sel_range, 0)
        bad_pixel(feature_dir, feature_vec[amt_0:amt_0+amt_1], poison_dir, sel_range, 1)
        bad_pixel(feature_dir, feature_vec[amt_0+amt_1:], poison_dir, sel_range, 2)
    else:
        bad_pixel(feature_dir, feature_vec, poison_dir, sel_range, bad_num, delta=delta)
    