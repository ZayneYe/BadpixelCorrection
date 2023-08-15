import os
import shutil

if __name__ == "__main__":
    path = '/data1/S7-ISP-Dataset'
    data_dir = '/data1/S7-ISP-Dataset/medium_dng'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cnt = 1
    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            if file.split('.')[0] == 'medium_exposure' and file.split('.')[1] == 'dng':
                new_file = f'{cnt}.dng'
                file_path = os.path.join(path, dir, file)
                shutil.copy(file_path, data_dir)
                shutil.move(os.path.join(data_dir, file), os.path.join(data_dir, new_file))
                cnt += 1