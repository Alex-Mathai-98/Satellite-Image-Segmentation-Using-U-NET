# Satellite Image Segmentation Using the U-Net Deep Learning Architecture
<p align="center"> <b>Alex Mathai(BITS Pilani)    |    Ashutosh Kumar Jha(ISRO)    |    Sameer Saran(ISRO)</b> </p>

# TL;DR
This project contains scripts of a 3 month project at the Indian Space Research Organisation, Dehradun. This script contains code for a U-Net deep learning model for segmentation of roads and water bodies in high-resolution satellite images. 

# Environment
To easily replicate my environment, please clone the **isro.yml** file using **Conda**.

# Data
The data for this project has been taken from the Kaggle competition "Can you train an eye in the sky ?", DSTL UK. 

### Explanation (Optional)
Most images that we capture on phones are made of 3-Bands - red, green and blue (RGB). However satellite images have multiple bands. The data in this competition had 20 bands divided into three categories. 

1. **GROUP I** : Panchromatic (Black and White) as well as RGB Bands. Often known as PRGB.
2. **GROUP II** : M-Spectrum (A collection of 8 bands from Coastal Blue to Near-Infrared 2)
3. **GROUP III** : A-Spectrum (A collection of 8 more bands) 

### Steps (Essential)
1. Download all data from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data, after accepting all terms and conditions.
2. Extract the data in the "three_band.zip" into the ```Data/three_band``` folder.
3. Extract the data in the "sixteen_band.zip" into the  ```Data/sixteen_band``` folders.
4. Extract the data in the "grid_sizes.csv.zip" and "train_wkt_v4.csv.zip" in the ```Data``` folder.

# Flow Chart
<p align="middle">
  <img src="/Images/project_flow_chart.png" width="600" />
</p>

# Preprocessing
Navigate to the main folder where all the scripts exist. 
And run the commands given below. If interested in understanding each process, please read through the optional explainations.

## Resizing and Panchromatic Sharpening of Images

### Explanation (Optional)
The 20 bands span a huge range of the electromagnetic spectrum. Because of this, if we were to compare the sizes of the image of a particular location across the 3 groups, they would vary drastically. **GROUP III** is the smallest, then comes **GROUP II** and the largest is **GROUP I**.

As deep learning models expect a fixed size input, we would need to resize **GROUP III** and **GROUP II** to the size of **GROUP I**. **GROUP III** being the smallest, suffers a drastic drop in image quality while resizing. Hence **GROUP III** was left out. 

**GROUP II** can be resized successfully but suffers from alignment issues. For example, the starting and the ending of roads did not exactly coincide for the resized **GROUP II** bands and the PRGB. Hence bringing alignment was necessary. This is done through a process called **Panchromatic Sharpening**.

### Run Commands (Essential)
```python
python pan_sharpen_M.py
```
```python
python pan_sharpen_RGB.py
```
```python
python bands_stack_creator.py
```

## Ground truth binary masks for images

### Explanation (Optional)
Although the original task was a multi-class prediction problem, we break this multiclass problem into individual binary-prediction problems. For example Road and Not-Road, Water and Not-Water.

### Run Commands (Essential)
```python
python mask_creator.py
```

# Segmenting Roads
You can either use my **Pre-trained Weights** to predict tarred roads from satellite images or you can **Train from Scratch**.

## Architecture
![U-Net](/Images/U-Net.png)

### Explanation (Optional)
The architecure used for this project is a deep U-Net model. In vanilla CNNs, the prediction of the deeper layers contain more semantic information, whereas the predictions from the earlier layers contain more spatial information. In simpler terms, the deeper layers answer the
question "What?", whereas the earlier layers answer the question "Where?" (Note that as we go from left to right we are in fact
going deeper into the network). This network however tries to combine both of these predictions through the red channels. The
left arm of the "U" contains the earlier layers that answer the question "Where in this picture?" whereas the right arm contains
the deeper layers that answer the question "What is this picture?”. It is pretty evident from the diagram, that the network
concatenates (joins) the deeper layers of the right arm with the earlier layers of the left arm. In doing so, it tries to merge the
"What" and the "Where" information - two questions that are essential to image segmentation.

## Train from Scratch 
In the ```Data_masks/Road``` folder segregate the pictures into training and testing. Place ```6100_2_2_Road.tif``` and ```6140_3_1_Road.tif``` 
in the test folder. Now delete all the other images except for ```6070_2_3_Road.tif```, ```6100_1_3_Road.tif```, ```6100_2_3_Road.tif```, ```6110_1_2_Road.tif```, ```6110_3_1_Road.tif```, ```6110_4_0_Road.tif```, ```6120_2_2_Road.tif``` and ```6140_1_2_Road.tif``` as they are the only images that have roads in them.

### Run Commands (Essential)

To create a train-dataset run the below command.
```python
python create_dataset_for_roads.py
```

To train the U-Net model to detect tarred roads, run the below command.
```python
python train_road_tar.py
```

To test the U-Net model on the test images, run the below command.
```python
python test_roads.py
```

## Use Pre-trained Weights
Download my pretrained weights from this [link](https://drive.google.com/drive/folders/1YGMZVCRn5UVlQL3V-MTwT4QHtI7HJJSX?usp=sharing). Place the weights in the ```Parameters/Road_tar``` folder. Please modify the first line of the ```checkpoint``` file so that your my absolute path is replaced by your absolute path.

To test the U-Net model on the test images, run the below command.
```python
python test_roads.py
```

## Results from Pre-trained Weights
We measure our performance by the **Jaccard Metric**. For tarred roads, we achieve a high Jaccard score of **0.6**.

<p align="middle">
  <img src="/Images/Road_Input.png" width="350" hspace="50" />
  <img src="/Images/Road_Output.png" width="350" /> 
</p>


# Segmenting Water bodies
Because of the lack of data for water bodies, deep learning models failed to converge on training. Hence we resorted to using an ensemble of machine learning models. 

## Architecture
<p align="middle">
  <img src="/Images/ensemble.png" width="600" />
</p>

## Train from Scratch
In the ```Data_masks/Fast_H20``` folder segregate the pictures into training and testing. Place ```6070_2_3_Fast_H20.tif``` in the ```Data_masks/Fast_H20/Test``` folder and keep ```6100_2_2_Fast_H20.tif``` as is. Delete the rest, as all other images do not have large water bodies in them.

### Run Commands (Essential)

To create a train-dataset run the below command.
```python
python create_dataset_for_fast_h20.py
```

To train the ensemble in detecting deepwater bodies run the below command.
```python
python train_h20.py
```

To test the ensemble model on the test images run the below command.
```python
 python test_h20.py
```

## Use Pre-trained Weights
Download my pretrained weights from this [link](https://drive.google.com/drive/folders/173-JAXabZmhCojRN_3UQK5VJPjVjAnwB?usp=sharing). Place the weights in the ```Parameters/Fast_H20``` folder.

To test the ensemble model on the test images run the below command.
```python
 python test_h20.py
```

## Results from Pre-trained Weights
We measure our performance by the **Jaccard Metric**. For water bodies, we achieve a high Jaccard score of **0.7**.

<p align="middle">
  <img src="/Images/Water_Input.png" width="350" hspace="50" />
  <img src="/Images/Water_Output.png" width="350" /> 
</p>





