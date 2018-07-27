#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:25:12 2018

@author: alex
"""

from gdal_utilities import gdal_utils
import numpy as np
import cv2
import os 

def get_image_names(imageId):
    '''
    Get the names of the tiff files with ID "imageId".
    '''
    
    inDir = os.getcwd()
    
    d = {'3': '{}/Data/pan_sharpened_images/{}_RGB.tif'.format(inDir, imageId),
         'A': '{}/Data/sixteen_band/{}_A.tif'.format(inDir, imageId),
         'M': '{}/Data/pan_sharpened_images/{}_M.tif'.format(inDir, imageId),
         'P': '{}/Data/sixteen_band/{}_P.tif'.format(inDir, imageId),
         }
    return d


def get_images(imageId, img_key = None):
    '''
    Load images correspoding to imageId

    Parameters
    ----------
    imageId : str
              imageId as used in grid_size.csv
    img_key : str.
              {None, '3', 'A', 'M', 'P'}, optional
              Specify this to load single image
              None loads all images (i.e. every band of the same image) and returns in a dict
              '3' loads image from three_band/
              'A' loads '_A' image from sixteen_band/
              'M' loads '_M' image from sixteen_band/
              'P' loads '_P' image from sixteen_band/

    Returns
    -------
    images : dict
             A dict of image data from TIFF files as numpy array
    '''
    
    creator = gdal_utils()
    
    img_names = get_image_names(imageId)
    images = dict()
    if img_key is None:
        for k in img_names.keys():
            images[k] = creator.gdal_to_nparr(img_names[k])
    else:
        images[img_key] = creator.gdal_to_nparr(img_names[img_key])
    return images

def compute_EVI(nir,image_r,image_b):
    '''Computes EVI.
    
    Description :
        evi = ( (nir - image_r)/(nir + 6*image_r - 7.5*image_b + 1) )*2.5

    Arguments :
        nir     -- np.array.
                   Near Infrared Band
        image_r -- np.array.
                   Red Band
        image_b -- np.array
                   Green Band
    Returns :
        evi -- np.array.
               Enhanced Vegetation Index
    '''

    # Enhanced Vegetation Index
    L = 1.0
    C1 = 6.0
    C2 = 7.5
    
    nir     = nir.astype(np.float64)
    image_r = image_r.astype(np.float64)
    image_b = image_b.astype(np.float64)
    
    evi = np.zeros(shape = nir.shape,dtype = np.float64)

    # Denominator of formula must not be zero
    x_s,y_s = np.where( ((nir + C1 * image_r - C2 * image_b + L) != 0.) )
    evi[x_s,y_s] = ( ( (nir[x_s,y_s] - image_r[x_s,y_s]) / (nir[x_s,y_s] + C1 * image_r[x_s,y_s] - C2 * image_b[x_s,y_s] + L) )*2.5 )*1023.5 + 1023.5 
    
    x_s,y_s = np.where( (nir + C1 * image_r - C2 * image_b + L) == 0.  )
    evi[x_s,y_s] = 0

    # Rescaling the values between 0 to 1   
    x_s,y_s = np.where(evi > 2047)
    evi[x_s,y_s] = 2047
    
    x_s,y_s = np.where(evi < 0)
    evi[x_s,y_s] = 0
        
    return evi 

def compute_NDVI(nir,image_r):
    ''' Computes NDVI.

    Description :

        NDVI = (NIR - RED)/(NIR + RED)

    Arguments :
        nir     -- np.array.
                   Near Infrared Band
        image_r -- np.array.
                   Red Band
    Returns :
        NDVI -- np.array.
                Normalized Difference Vegetation Index
    '''
    
    # Normaliized Difference Vegetation Index
    
    ndvi = np.zeros(shape = nir.shape,dtype = np.float64)
    
    nir = nir.astype(np.float64)
    image_r = image_r.astype(np.float64)
    
    # Denominator should not be zero
    x_s,y_s = np.where( (nir + image_r) != 0.  )
    ndvi[x_s,y_s] = ((nir[x_s,y_s] - image_r[x_s,y_s]) / (image_r[x_s,y_s] + nir[x_s,y_s]))*1023.5 + 1023.5
    
    x_s,y_s = np.where( (nir + image_r) == 0.  )
    ndvi[x_s,y_s] = 0
    
    # Rescaling the values between 0 to 1
    x_s,y_s = np.where(ndvi > 2047)
    ndvi[x_s,y_s] = 2047
    
    x_s,y_s = np.where(ndvi < 0)
    ndvi[x_s,y_s] = 0
    
    return ndvi

def compute_NDWI(nir,image_g):
    ''' Computes NDWI.

    Description :

        NDVI = (NIR - GREEN)/(NIR + GREEN)

    Arguments :
        nir     -- np.array.
                   Near Infrared Band
        image_g -- np.array.
                   Green Band
    Returns :
        NDWI -- np.array.
                Normalized Difference Water Index
    '''
    
    # Normalized Difference Water Index
    ndwi = np.zeros(shape = nir.shape,dtype = np.float64)
    
    nir = nir.astype(np.float64)
    image_g = image_g.astype(np.float64)
    
    # Denominator should not be zero
    x_s,y_s = np.where( (nir + image_g) != 0 )
    ndwi[x_s,y_s] = ( (image_g[x_s,y_s]) - (nir[x_s,y_s]) ) / ( (image_g[x_s,y_s]) + nir[x_s,y_s] )*1023.5 + 1023.5
    
    x_s,y_s = np.where( (nir + image_g) == 0 )
    ndwi[x_s,y_s] = 0
    
    # Rescaling the values between 0 to 1
    x_s,y_s = np.where(ndwi > 2047)
    ndwi[x_s,y_s] = 2047
    
    x_s,y_s = np.where(ndwi < 0)
    ndwi[x_s,y_s] = 0
    
    return ndwi

def compute_CCCI(nir,re,image_r):

	''' Computes Canopy Chlorophyll Content Index.

	Description :
		CCCI = ( (NIR - RED_EDGE)/(NIR + RED_EDGE) )/( (NIR - RED)/(NIR + RED) )

	Arguments :
		nir     -- np.array.
                   Near Infrared Band
        re      -- np.array.
                   Red Edge
        image_r -- np.array.
                   Red Band
    Returns :
        CCCI -- np.array.
                Canopy Chlorophyll Content Index
	'''
	ccci = np.zeros(shape = nir.shape,dtype = np.float64)
    
	nir = nir.astype(np.float64)
	re =  re.astype(np.float64)
	image_r = image_r.astype(np.float64)
    
	# Denominator should not be 0  
	x_s,y_s = np.where( ((nir + re) == 0) | ((nir - image_r) == 0) | ((nir + image_r) == 0) )  
	ccci[x_s,y_s] = 0
    
	x_s,y_s = np.where( ((nir + re) != 0) & ((nir - image_r) != 0) & ((nir + image_r) != 0) )
	ccci[x_s,y_s] = ( ((nir[x_s,y_s] - re[x_s,y_s]) / (nir[x_s,y_s] + re[x_s,y_s]) ) /( (nir[x_s,y_s] - image_r[x_s,y_s]) / (nir[x_s,y_s] + image_r[x_s,y_s]) ) )*1023.5 + 1023.5
    
	# Rescaling the values between 0 and 1
	x_s,y_s = np.where(ccci > 2047)
	ccci[x_s,y_s] = 2047
    
	x_s,y_s = np.where(ccci < 0)
	ccci[x_s,y_s] = 0
    
	return ccci

def compute_SAVI(nir,image_r):
    ''' Soil Adjusted Vegetation Index.

    Description :

        SAVI = (NIR - RED)/(NIR + RED)

    Arguments :
        nir     -- np.array.
                   Near Infrared Band
        image_r -- np.array.
                   Red Band
    Returns :
        savi -- np.array.
                Soil Adjusted Vegetation Index
    '''

    savi = np.zeros(shape = nir.shape,dtype = np.float64)
    
    nir = nir.astype(np.float64)
    image_r = image_r.astype(np.float64)
        
    x_s,y_s = np.where( (nir + image_r) != 0 )
    savi[x_s,y_s] = ((nir[x_s,y_s] - image_r[x_s,y_s])/(nir[x_s,y_s] +image_r[x_s,y_s]))*1023.5 + 1023.5
    
    x_s,y_s = np.where( (nir + image_r) == 0 )
    savi[x_s,y_s] = 0
    
    x_s,y_s = np.where( savi > 2047 )
    savi[x_s,y_s] = 2047
    
    x_s,y_s = np.where( savi < 0 )
    savi[x_s,y_s] = 0    
    
    return savi

def give_image_sandwhich(imageId):
    ''' Returns the image stack of 14 channels. '''
    
    # np.tranpose permutes the axes of an n-dim array

    # Get all images of a Id
    images =  get_images(imageId,None)
    
    print(images['P'].shape)
    print(images['M'].shape)
    print(images['3'].shape)

    # Get the pansharpened images    
    img_P = images['P'] # P band    
    img_rgb = images['3'] # 3 band      
    img_m = images['M'] # M band
    
    images = None
                  
    # Calculating all_indices
    image_r = img_rgb[:,:,0]
    image_g = img_rgb[:,:,1]
    image_b = img_rgb[:,:,2]

    nir = img_m[:,:,6]
    re  = img_m[:,:,5]
    
    # Enhanced vegetation Index
    evi  = np.expand_dims(compute_EVI(nir,image_r,image_b),2) # Done 
    
    # Normalized Difference Water Index        
    ndwi = np.expand_dims(compute_NDWI(nir,image_g), 2) # Done
    
    # Normalized Difference Vegetation Index    
    ndvi = np.expand_dims(compute_NDVI(nir,image_r), 2) # Done
    
    # Canopy Chlorophyll Content Index
    ccci = np.expand_dims(compute_CCCI(nir,re,image_r),2)       
    
    # Soil Adjusted Vegetation Index
    savi = np.expand_dims(compute_SAVI(nir,image_r), 2)
    
    # exclude the red, green and blue channels of the M-band    
    img_m_concs = np.concatenate((np.expand_dims(img_m[:,:,0],2),np.expand_dims(img_m[:,:,3],2),np.expand_dims(img_m[:,:,5],2),np.expand_dims(img_m[:,:,6],2),np.expand_dims(img_m[:,:,7],2)),axis = 2)
    
    new_img = np.concatenate((img_P,img_rgb,img_m_concs,evi,ndwi,ndvi,savi,ccci), axis = 2)
    
    return new_img

if __name__ == '__main__':
    
    import pandas as pd

    
    # Base Address
    inDir = os.getcwd()
        
    # train-images multiploygon co-ordinates
    df = pd.read_csv(os.path.join(inDir,'Data/train_wkt_v4.csv'))
    print(df.head())
    
    # Distinct imageIds in the DataFrame    
    trainImageIds = []
        
    for i in range(len(df)):
        string = (df.iloc[i,0])
        
        if string not in trainImageIds:
           trainImageIds.append(string) 
               
    print("The are " + str(len(trainImageIds)) + " distinct ids." + str(df.head()))
    trainImageIds.sort()
    
    creator = gdal_utils()         
    for imageId in trainImageIds :
        new_img = give_image_sandwhich(imageId)
        creator.create_tiff_file_from_array("./Data/sixteen_band/" + imageId + "_P.tif","./Data/image_stacks/" + imageId + ".tif",new_img)













