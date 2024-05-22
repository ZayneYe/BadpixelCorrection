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


# save numpy files for original .dng images
# import rawpy
# import os
# import numpy as np
# org_dir = '/data1/Invertible_ISP/original_images'
# save_dir = '/data1/Bad_Pixel_Detection/FiveK/original_images'
# for i, dng in enumerate(os.listdir(org_dir)):
#      file_name = dng.split('.')[0]
#      raw = rawpy.imread(os.path.join(org_dir, dng))
#      raw_data = raw.raw_image
#      raw_data = np.asarray(raw_data)
#      np.save(os.path.join(save_dir, file_name), raw_data)