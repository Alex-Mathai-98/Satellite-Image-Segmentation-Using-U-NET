# Satellite Image Segmentation Using the U-Net Deep Learning Architecture
<p align="center"> <b>Alex Mathai(BITS Pilani)    |    Ashutosh Kumar Jha(ISRO)    |    Sameer Saran(ISRO)</b> </p>

# TL;DR
This project contains scripts of a 3 month project at the Indian Space Research Organisation, Dehradun. This script contains code for a U-Net deep learning model for segmentation of roads and water bodies in high-resolution satellite images. 

# Replicating the Environment
To easily replicate my environment, please clone the **isro.yml** file using **Conda**.

# Data
The data for this project has been taken from the Kaggle competition "Can you train an eye in the sky ?", DSTL UK. 

## Explanation (Optional)
Most images that we capture on phones are made of 3-Bands - red, green and blue (RGB). However satellite images have multiple bands. The data in this competition had 20 bands. 

1. Panchromatic (Black and White) as well as RGB Bands. Often known as PRGB.
2. M-Spectrum (A collection of 8 bands from Coastal Blue to Near-Infrared 2)
3. A-Spectrum (A collection of 8 more bands) 

## Step 1
Download all data from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data, after accepting all terms and conditions.

## Step 2
Extract the data in the "three_band.zip" into the Data/three_band folder.
Extract the data in the "sixteen_band.zip" into the  Data/sixteen_band folders.
Extract the data in the "grid_sizes.csv.zip" and "train_wkt_v4.csv.zip" in the Data folder.

# Project Flow Chart
![Flow Chart](/Images/project_flow_chart.png)

# Preprocessing
Navigate to the main folder where all the scripts exist. 
And run the commands given below. If interested in understanding each process, please see deta

## Resizing and Pansharpening of Images

### Explanation (Optional)
Because there are 20 bands being used, the sizes of the image of the same location are different across different bands. The M-Spectrum and the A-Spectrum are smaller than the Panchromatic and RGB. Hence both the A and M Spectrums are resized to the size of the 


```python
"python pan_sharpen_M.py"
```
```python
"python pan_sharpen_RGB.py"
```
```python
"python bands_stack_creator.py"
```
4. "python mask_creator.py"

// For Roads //

**Step 5:**
In the Data_masks/Road folder segregate the pictures into training and testing. Place "6100_2_2_Road.tif" and "6140_3_1_Road.tif" 
in the test folder and keep "6070_2_3_Road.tif", "6100_1_3_Road.tif", "6100_2_3_Road.tif", "6110_1_2_Road.tif", "6110_3_1_Road.tif", "6110_4_0_Road.tif", "6120_2_2_Road.tif" and "6140_1_2_Road.tif" as is. Delete the rest as all other images do not have roads in them.

Now run "python create_dataset_for_roads.py"

**Step 6:**
To train the U-Net model to detect tarred roads, run "python train_road_tar.py"

**Step 7:**
To test the U-Net model on the test images, run "python test_roads.py"

// For Water //

**Step 5:**
In the Data_masks/Fast_H20 folder segregate the pictures into training and testing. Place "6070_2_3_Fast_H20.tif" in the Data_masks/Fast_H20/Test
folder and keep "6100_2_2_Fast_H20.tif" as is.Delete the rest as all other images do not have large water bodies in them.

Now run "python create_dataset_for_fast_h20.py"

**Step 6:**
To train the U-Net model to detect tarred roads, run "python train_h20.py"

**Step 7:**
To test the U-Net model on the test images, run "python test_h20.py"










