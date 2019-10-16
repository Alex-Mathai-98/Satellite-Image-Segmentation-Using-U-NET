#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:39:56 2018

@author: alex
"""

from gdal_utilities import gdal_utils
import numpy as np
import os
from imgaug import augmenters as iaa
import pickle

def get_masks_list():
    
    '''
    Gets the list of masked tiff files intended for training and testing 
    '''
    
    inDir = os.getcwd()    
    
    files_train_temp = os.listdir(os.path.join(inDir,"Data_masks/Road"))
    files_train_final = []
    
    i = 0 
    for file in files_train_temp : 
        
        extension = os.path.splitext(file) 
    
        if extension[0] == 'Test' :
            continue
        else:
            files_train_final.append(file[:-9])
    
        i += 1
    
    
    files_test_temp = os.listdir( os.path.join(inDir,"Data_masks/Road/Test") )
    files_test_final = []
    
    for file in files_test_temp:
        files_test_final.append(file[:-9])
    
    return files_train_final,files_test_final


def get_training_image_pair(files_train,imageId):
    ''' Gets the input,truth for an image.

    Description :
        Get image-ground-truth pair of the image with id "imageId" from the list of
        training images "files_train"

    Arguments :
        files_train -- List.
                       The list of names of the tiff images that will be used for training
        imageId     -- String.
                       The id of the image.
                       ex. '6010_0_0'
    Returns :
        (image_train,truth) -- Tuple.
                               This tuple containing the input image and the ground truth.
    '''
    
    if imageId not in files_train:
        raise ValueError("Invalid value of imageId")
        return None

    # Using gdal to read the input image
    reader = gdal_utils()
    path = os.path.join(os.getcwd(),"Data/image_stacks/" + imageId + ".tif")
    image_train = reader.gdal_to_nparr(path)
    
    if image_train is None:
        print("Failed to load image. Wrong path name provided.")
        return None

    # Using gdal to read the ground truth
    path = os.path.join(os.getcwd(),"Data_masks/Road/" + imageId + "_Road.tif")
    truth = reader.gdal_to_nparr(path)
    
    if truth is None:
        print("Failed to load groung truth. Wrong path name provided.")
        return None
    
    return (image_train,truth)

def create_list_of_augmenters(flip_vertical = True, flip_horizontal = True,random_rotations = True):
    '''Creates a list of image augmenters.

    Description :
        Creates a list of image augmenters that can possibly flip an image vertically,
        horizontally and perform clockwise rotations in increments of 90 degrees out
        of [90,180,270,360].

    Arguments :
        flip_vertical    -- Bool.
                            The augmenters created will have the ability to flip
                            images vertically.
        flip_horizontal  -- Bool.
                            The augmenters created will have the ability to flip
                            images horizontally.
        random_rotations -- Bool.
                            The augmenters created will have the ability to rotate
                            images by 90,180,270 and 360 degrees.

    Returns :
        ans -- List.
               The list contains a number of image augmenters that have the capability to perform
               flips and rotations as decided by the input parameters "flip_vertical",
               "flip_horizontal" and "random_rotations". If all three parameters are "false"
               then "None" is returned.
    '''
    
    if flip_vertical and flip_horizontal and random_rotations :
        
        ans = {}

        # flip up down
        flip_ud = 1.0

        # flip left right
        flip_lr = 1.0

        # Add 4 augmenters with different rotation capabilities.
        for i in range(4):
             
            string = str(i)
            
            ans[string] = iaa.Sequential([
                                iaa.Flipud(flip_ud),
                                iaa.Fliplr(flip_lr),
                                iaa.Affine(rotate = i*90,cval = 0),
                            ])
        return ans
    
    elif flip_vertical and flip_horizontal and (not random_rotations) :
        
        ans = {}

        # flip up down
        flip_ud = 1.0

        # flip left right
        flip_lr = 1.0

        ans["0"] = iaa.Sequential([
                            iaa.Flipud(flip_ud),
                            iaa.Fliplr(flip_lr),
                        ])
    
        return ans
    
    elif flip_vertical and (not flip_horizontal) and (not random_rotations) :
        
        ans = {}

        # flip up down
        flip_ud = 1.0
        
        ans["0"] = iaa.Sequential([
                            iaa.Flipud(flip_ud),
                        ])
    
        return ans

    elif  flip_vertical and (not flip_horizontal) and random_rotations :

        ans = {}

        # flip up down
        flip_ud = 1.0

        # Add 4 augmenters with different rotation capabilities.
        for i in range(4):
            
            string = str(i)
            
            ans[string] = iaa.Sequential([
                                iaa.Flipud(flip_ud),
                                iaa.Affine(rotate = i*90,cval = 0),
                            ])
        
        return ans        
        
    elif (not flip_vertical) and flip_horizontal and (not random_rotations) :
        
        ans = {}

        # flip left right
        flip_lr = 1.0
    
        ans["0"] = iaa.Sequential([
                                iaa.Fliplr(1.0),
                ])
        
        return ans
    
    elif (not flip_vertical) and flip_horizontal and random_rotations :
        
        ans = {}

        # flip left right
        flip_lr = 1.0

        # Add 4 augmenters with different rotation capabilities.
        for i in range(4) :
            
            string = str(i)
            
            ans[string] = iaa.Sequential([
                                    iaa.Fliplr(1.0),
                                    iaa.Affine(rotate = i*90,cval = 0),
                                ])
        
        return ans
    
    elif (not flip_vertical) and (not flip_horizontal) and random_rotations :
        
        ans = {}

        # Add 4 augmenters with different rotation capabilities.
        for i in range(4):
            
            string = str(i)
            
            ans[string] = iaa.Sequential([
                                    iaa.Affine(rotate = i*90,cval = 0),
                    ])
    
    
        return ans
    
    else:
        return None
    
        
def data_augmentation(arr_input,arr_truth,flip_vertical = True, flip_horizontal = True,random_rotations = True):
    """ Augments the arrays "arr_input" and "arr_truth".

    Description :
        Augments "arr_input" and "arr_truth" by the parameters provided.
        It can rotate and flip images randomly.

    Arguments :
        flip_vertical    -- Bool.
                            "arr_input" will be flipped vertically.
        flip_horizontal  -- Bool.
                            "arr_input" will be flipped horizontally.
        random_rotations -- Bool.
                            "arr_input" will be rotated clockwise by
                            either 90,180,270 or 360 degees.The degree
                            of rotation will be decided randomly.

        If "flip_vertical" and "flip_horizontal" are "True", then image will be flipped
        vertically first and then flipped horizontally.

    Returns :
        arr_input,arr_truth -- The augmented input array ("arr_input") and the augmented truth array
                               ("arr_truth").
    """
                   
    augmenter_list = create_list_of_augmenters(flip_vertical,flip_horizontal,random_rotations) 

    # If "flip_vertical","flip_horizontal" and "random_rotations" are all "False"
    if augmenter_list is None:
        
        return arr_input,arr_truth
    
    else :

        # Random number to select randomly one augmenter from the list
        num = np.random.randint(0,len(augmenter_list))
        
        augmen = augmenter_list[str(num)]
        
        arr_input = augmen.augment_image(arr_input)
        arr_truth = augmen.augment_image(arr_truth)
        
        return arr_input,arr_truth


def create_training_batch(file_train,cover = 0.1,batch_size = 16,img_rows = 112,img_cols = 112,num_channels = 14):
   """ Creates a training batch with dimenions (batch_size,img_rows,img_cols,num_channels).

   Description :
       Given "file_train" (the list of training image file Ids), a mini batch of
       data is created ensuring that coverage of a certain class is >= "cover" for
       each image in the minibatch.

   Arguments :
       file_train   -- List.
                       The list of training Image File Ids.
       cover        -- Float. (Default = 0.1)
                       The minimum fraction of pixels in an image covered by a certian class.
       batch_size   -- Int.   (Default = 16)
                       The number of images in a mini-batch.
       img_rows     -- Int.   (Default = 112)
                       The number of rows (height) of an image to be extracted.
       img_cols     -- Int.   (Default = 112)
                       The number of cols (width) of an image to be extracted.
       num_channels -- Int.   (Default = 14)
                       The depth of an image in a mini-batch.

   Returns :
       X -- A minibatch of training patches.
       Y -- A minibatch of corresponding ground truth patches. 
   """

   # X - array that stores all the inputs
   # Y - array that stores all the ground truths   
   X = np.zeros(shape = (batch_size,img_rows,img_cols,num_channels),dtype = np.float64)
   Y = np.zeros(shape = (batch_size,img_rows,img_cols,1),dtype = np.float64)
   
   size = len(file_train)

   # Select 4 random images from the list
   number1 = np.random.randint(0,size)
   number2 = np.random.randint(0,size)
   number3 = np.random.randint(0,size)
   number4 = np.random.randint(0,size)
   
   file1 = file_train[number1]
   file2 = file_train[number2]
   file3 = file_train[number3]
   file4 = file_train[number4]
      
   image1,truth1 = get_training_image_pair(file_train,file1)
   image2,truth2 = get_training_image_pair(file_train,file2)
   image3,truth3 = get_training_image_pair(file_train,file3)
   image4,truth4 = get_training_image_pair(file_train,file4)   
   
   list_img   = [image1,image2,image3,image4]
   list_truth = [truth1,truth2,truth3,truth4]

   # Find indices where roads are present.   
   x_1s,y_1s,z_1s = np.where(truth1 == 255)
   x_2s,y_2s,z_2s = np.where(truth2 == 255)
   x_3s,y_3s,z_3s = np.where(truth3 == 255)
   x_4s,y_4s,z_4s = np.where(truth4 == 255)

   pixel_cords = [(x_1s,y_1s,z_1s),(x_2s,y_2s,z_2s),(x_3s,y_3s,z_3s),(x_4s,y_4s,z_4s)]

   # Total batch size is 16.
   percentage = 0
   for i in range(batch_size) :

       counting = 0

       # Finding the image with atleast coverage >= "cover"       
       while percentage < cover :

            # select one training image randomly           
            select = np.random.randint(0,len(list_img))

            selected_img   = list_img[select]
            selected_truth = list_truth[select]
            (x_s,y_s,z_s) = pixel_cords[select] 

            # if the image contains no water, then skip
            # else choose one point from the image.            
            if len(x_s) != 0 :
                random = np.random.randint(0,len(x_s))
            else:
                counting += 1

                # If the 4 images chosen seem to have very less road coverage
                # such that the mini-batch cannot be made, then choose 4 other
                # images.
                if(counting == 100) :
                    print("Unlucky Draw !")
                    image1 = None
                    image2 = None
                    image3 = None 
                    image4 = None
                    truth1 = None
                    truth2 = None
                    truth3 = None
                    truth4 = None 
                    return create_training_batch(file_train,cover,batch_size,img_rows,img_cols,num_channels)
                
                continue

            # Try to create a patch with point "select" as
            # the top left corner and check for coverage.           

            start_col = y_s[random]
            end_col = start_col + img_cols - 1

            # dimensions exceeded            
            if end_col >= selected_truth.shape[1] :
                counting += 1
                
                if(counting == 100) :
                    print("Unlucky Draw !")
                    image1 = None
                    image2 = None
                    image3 = None 
                    image4 = None
                    truth1 = None
                    truth2 = None
                    truth3 = None
                    truth4 = None 
                    return create_training_batch(file_train,cover,batch_size,img_rows,img_cols,num_channels)
                
                continue
            
            start_row = x_s[random]
            end_row = start_row + img_rows -1

            # dimensions exceeded
            if end_row >= selected_truth.shape[0] :
                counting += 1
                
                if(counting == 100) :
                    print("Unlucky Draw !")
                    image1 = None
                    image2 = None
                    image3 = None 
                    image4 = None
                    truth1 = None
                    truth2 = None
                    truth3 = None
                    truth4 = None 
                    return create_training_batch(file_train,cover,batch_size,img_rows,img_cols,num_channels)
                
                
                continue

            coverage = np.sum((selected_truth[start_row:end_row + 1,start_col:end_col + 1,:])/255)/(img_cols*img_rows) 
            if  coverage >= cover :
                arr_input = selected_img[start_row:end_row + 1,start_col:end_col + 1,:]
                arr_truth = (selected_truth[start_row:end_row + 1,start_col:end_col + 1,:])/255

                percentage = np.sum(arr_truth)/(img_cols*img_rows)
           
            counting += 1
       
       # Making random choices regarding horizontal flips,vertical flips and rotations.           
       choice_1 = [True,False]
       choice_2 = [True,False]
       choice_3 = [True,False]
       
       deci_1 = np.random.randint(0,2)
       deci_2 = np.random.randint(0,2)
       deci_3 = np.random.randint(0,2)
       
 
       #print("{} {} {}".format(choice_1[deci_1],choice_2[deci_2],choice_3[deci_3]))

       # Do data Augmentation
       arr_input2,arr_truth2 = data_augmentation(arr_input,arr_truth,choice_1[deci_1],choice_2[deci_2],choice_3[deci_3])
       
       assert( arr_input.shape == arr_input2.shape )
       assert( arr_truth.shape == arr_truth2.shape )

       # Free Memory       
       arr_input = None
       arr_truth = None
       
       X[i],Y[i] = arr_input2,arr_truth2
       
       assert(X[i].shape == (img_rows,img_cols,num_channels))
       assert(Y[i].shape == (img_rows,img_cols,1))

       percentage = 0

   # Assert Conditions   
   assert(X.shape == (batch_size,img_rows,img_cols,num_channels))
   assert(Y.shape == (batch_size,img_rows,img_cols,1))

   # Free Memory
   image1 = None
   image2 = None
   image3 = None 
   image4 = None
   truth1 = None
   truth2 = None
   truth3 = None
   truth4 = None 
    
   return X,Y


def save_data(X,Y,i):
    ''' Saves the mini-batch to disk.

    Arguments :
         X -- np.array.
              The input mini-batch.
         Y -- np.array.
              The corresponding ground truths of the images of the mini-batch.
         i -- Int.
              The mini-batch number.
    '''
    
    with open("./Data_training/Road/train" + "_" + str(i) + "_" + ".pkl","wb") as f:
        pickle.dump(X,f)
    
    with open("./Data_training/Road/test" + "_" + str(i) + "_" + ".pkl","wb") as f:
        pickle.dump(Y,f)

if __name__ == '__main__' :
    
    file_train,file_test = get_masks_list()

    batch_size = 16
    img_rows = 112
    img_cols = 112
    num_channels = 14
    
    # mud roads
    file_train_mud = ['6140_1_2','6120_2_2','6110_1_2','6070_2_3','6100_2_3']
    
    # tarred roads
    for ids in file_train_mud : 
        file_train.remove(ids)
    
    file_train_tar = file_train
    
    cover = 0.1
    
    for i in range(1500) :
        X,Y = create_training_batch(file_train_tar,cover,batch_size,img_rows,img_cols,num_channels)
        save_data(X,Y,i)
        print(i)
        
    for k in range(1500,2000) : 
        X,Y = create_training_batch(file_train_mud,cover,batch_size,img_rows,img_cols,num_channels)
        save_data(X,Y,k)
        print(k)        
        


























