# Bad Pixel Correction

This code is for correction of the central pixel of a nxn patch using a 2-layer MLP

Dataset Preparation
```
1. Download Dataset
Samsung S7 ISP Dataset: https://www.kaggle.com/datasets/knn165897/s7-isp-dataset

2. Extract .dng images with medium exposure
cd scripts/
python create_dataset.py

3. Extract patches from each image
python crop_matrix.py
cut_size -> size of each patch
sample_amt -> number of patches to be extracted 

4. Split cropped patches into train, validation and test sets
python split_dataset.py 

4. Create a train set with multiple bad pixels
python poison_data.py
feature_dir -> folder containing cropped patches
bad_num -> number of neighboring bad pixels in each patch

5. Bad pixel injection into test images only for testing purposes
python bad_pixels.py
```

Training
```
** MLP**
Train on patches with no bad pixels in the neighborhood:
python train.py

Train on patches with one or more neighboring bad pixels
cd scripts/
python poison_data.py
python train.py --use_poison

**ViT AE**
python train_mae.py
```

Testing
```
Test on patches with no neighboring bad pixels
python test.py --mode test

Test on patches with multiple neighboring bad pixels
python test.py --mode corrupt
```