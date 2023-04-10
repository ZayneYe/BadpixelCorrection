import random
import os
import shutil

def move_data(dir, set, cate):
    dir_ = os.path.join(dir, cate)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    for file in set:
        shutil.move(os.path.join(dir, file), dir_)

if __name__ == "__main__":
    feature_dir = '../data/medium1_68750/feature_5'
    data = list(os.listdir(feature_dir))
    random.seed(77)
    random.shuffle(data)
    num = len(data)
    train_rate, val_rate = 0.8, 0.1
    train_num = int(num * train_rate)
    val_num = int(num * val_rate)
    
    train_set = data[:train_num]
    val_set = data[train_num:train_num + val_num]
    test_set = data[train_num + val_num:]
    
    move_data(feature_dir, train_set, 'train')
    move_data(feature_dir, val_set, 'val')
    move_data(feature_dir, test_set, 'test')