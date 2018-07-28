# Satellite-Image-Segmentation-Using-U-NET

**Code for MULTISPECTRAL GEOSPATIAL FEATURE EXTRACTION USING THE U-NET DEEP LEARNING ARCHITECTURE**  
**BY**
**Alex Mathai(BITS PILANI),**
**A. K. Jha(IIRS),**
**Sameer Saran(IIRS)**

The scripts of a 2 month project at the Indian Space Research Organisation,Dehradun. The scripts contain code for a U-Net model for Satellite Image Segmentation. 

The data for this project has been taken from the Kaggle competition "Can you train an eye in the sky ?", DSTL UK. The model given in the scripts will work for the road and water categories.

**Step 1 :**
Download all data from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data. After accepting all terms and
conditions.

**Step 2:**
Place the data inside "three_band.zip", "sixteen_band.zip" in the Data/three_band and Data/sixteen_band folders. Extract
"grid_sizes.csv.zip" and "train_wkt_v4.csv.zip" in the Data folder.

**Step 3:**
Create a conda environment using the "isro.yml" file and activate that environment.

**Step 4:**
Navigate to the main folder where all the scripts exist. Now run the following commands from the terminal.

1. "python pan_sharpen_M.py"
2. "python pan_sharpen_RGB.py"
3. "python bands_stack_creator.py"
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

Now run "create_dataset_for_fast_h20.py"

**Step 6:**
To train the U-Net model to detect tarred roads, run "python train_h20.py"

**Step 7:**
To test the U-Net model on the test images, run "python test_h20.py"










