import os
import shutil

path = "/data1/SIDD/DNG"
path_ = "/data1/SIDD/DNG_"

orders = []
for i, dng in enumerate(sorted(os.listdir(path))):
    order = dng.split('_')[0]
    if i == 0:
        shutil.copyfile(os.path.join(path, dng), os.path.join(path_, dng))
        orders.append(order)
    if order == orders[0]:
        continue
    else:
        orders.pop()
        orders.append(order)
        print(order)
        shutil.copyfile(os.path.join(path, dng), os.path.join(path_, dng))
    