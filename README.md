# Bad Pixel Correction

For correction of bad pixels, we propose two different approaches to deal with different error rates. First, we propose a patch-based correction approach using 2-layer MLP, where a $n\times n$ patch around the detected bad pixel is extracted, and passed through the correction network to obtain the actual value of the erroneous central pixel. While this method performs reasonably well for low error rates, it fails when the bad pixels are clustered or the number of bad pixels in a patch is very high. For this, we propose a Vision Transformer based Autoencoder (ViT AE) for pixel correction using complete image reconstruction.

[link to paper](https://arxiv.org/pdf/2402.05521.pdf)

**Dataset Preparation**
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

**Training**

MLP
```
Train on patches with no bad pixels in the neighborhood:
python train.py

Train on patches with one or more neighboring bad pixels
cd scripts/
python poison_data.py
python train.py --use_poison
```

ViT AE
```
python train_mae.py
```

**Testing**
```
Test on patches with no neighboring bad pixels
python test.py --mode test

Test on patches with multiple neighboring bad pixels
python test.py --mode corrupt
```

## Citation
If you find this repo useful for your research, please consider citing the following work:
```
@InProceedings{sarkar_2024_CVPRW,
    author       = {Sarkar, Sreetama and Ye, Xinan and Datta, Gourav and Beerel, Peter},
    title        = {FixPix: Fixing Bad Pixels using Deep Learning}, 
    eprint       = {2310.11637},
    archivePrefix={arXiv},
    primaryClass ={eess.IV},
    year         = {2024}
}
```