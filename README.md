# SamsungPixelCaculation
 
This is the baseline network for predict the middle pixel value by pixels around it for the Samsung In-Pixel Computing project.

The folder hierarchy for the dataset should look like this:

<img width="95" alt="1678766420624" src="https://user-images.githubusercontent.com/106359260/224889848-bf2e552b-e403-42a9-8892-2e8fe63519e8.png">


where the S7-ISP-Dataset is download from https://www.kaggle.com/datasets/knn165897/s7-isp-dataset.

python create_dataset.py: to generate medium_dng
python bad_pixels.py: to generate medium
python split_dataset.py: to split data to train/validation/test

The algorithom to crop matrixs from images look like this:

![IMG_0360(20230313-211722)](https://user-images.githubusercontent.com/106359260/224891980-382cf691-0b90-4fc4-8fc4-7e8012d5d8cd.PNG)


To train:
python train.py

To evaluate:
python test.py

Here's a demo's results:

LR_curve.png

This demo used 2720 croped images from the original 110 images in .dng format training for 50 epochs.
