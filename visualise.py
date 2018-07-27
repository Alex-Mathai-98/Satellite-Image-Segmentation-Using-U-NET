#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:43:42 2018

@author: alex
"""
from gdal_utilities import gdal_utils
import pandas as pd
import numpy as np
import os
from shapely import affinity
from shapely.wkt import loads as wkt_loads
from matplotlib.patches import Polygon, Patch

# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt


'''
### See what a WTK instance looks like ###

polygonsList = {}
df_image = df[df.ImageId == '6060_2_3']
for cType in CLASSES.keys():
    polygonsList[cType] = wkt_loads(df_image[df_image.ClassType == cType].MultipolygonWKT.values[0])    
    
print(polygonsList[1])    
'''   

'''
#### WORKFLOW ####

1) Define your canvas i.e fig (MATPLOTLIB.figure)
2) Create subplots and get the axes instances i.e ax (MATPLOTLIB.add_subplot)
3) Get a WTK object instance that has sets of co-ordinates
3) Create 2-D Shapely object instance by judging what shape the WTK object is and passing the WTK instance to the correct constructor
4) Create a pacth by passing that 2-D object instance to the PolygonPatch class constructor of descartes
5) Adding that patch to the list of patches of the axis attribute
6) Plot that patch
7) Enjoy !!

    
from shapely.geometry import LineString
fig = plt.figure()
ax = fig.add_subplot(121) # Adding a subplot 
line =  LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)]) # Creating a line instance by passing a WTK instance
dilated = line.buffer(0.5)

patch1 = PolygonPatch(dilated,facecolor='#99ccff',edgecolor='#6699cc')
ax.add_patch(patch1)

x,y = line.xy
ax.plot(x,y,color = '#999999')
ax.set_xlim(-1,4)
ax.set_ylim(-1,3)

'''



# Give short names, sensible colors and zorders to object types
CLASSES = {
        1 : 'Bldg',
        2 : 'Struct',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Fast H20',
        8 : 'Slow H20',
        9 : 'Truck',
        10 : 'Car',
        }
COLORS = {
        1 : '0.7',
        2 : '0.4',
        3 : '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }

# Artists with lower zorder values are drawn first.
# Hence tracks are given the lowest order and cars have been given the highest order

ZORDER = {
        1 : 5,
        2 : 5,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        }

def get_distinct_id_nos(df):
    ''' Gets the Distinct Id Nos from the dataframe df. '''

    ans = []
    
    for i in range(len(df)):
        string = ((df.iloc[i,0])[:-4])
        
        if string not in ans:
           ans.append(string) 
           
        
    print("There are " + str(len(ans)) + " distinct ids in " + str(df.head()))
     
    ans.sort()
    
    return ans

def get_distinct_ids(df):
    ''' Returns a list of distinct Ids which are sorted from the pandas dataframe. '''

    ans = []

    for i in range(len(df)):
        string = (df.iloc[i,0])
        
        if string not in ans:
           ans.append(string) 
           
        
    print("The are " + str(len(ans)) + " distinct ids." + str(df.head()))

    ans.sort()
     
    return ans

def get_image_names(imageId):
    '''
    Get the names of the tiff files
    '''

    # Current Directory
    inDir = os.getcwd()

    d = {'3': '{}/Data/three_band/{}.tif'.format(inDir, imageId),
         'A': '{}/Data/sixteen_band/{}_A.tif'.format(inDir, imageId),
         'M': '{}/Data/sixteen_band/{}_M.tif'.format(inDir, imageId),
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
    img_key : {None, '3', 'A', 'M', 'P'}, optional
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
    
    reader = gdal_utils()
    img_names = get_image_names(imageId)
    images = dict()
    if img_key is None:
        for k in img_names.keys():
            images[k] = reader.gdal_to_nparr(img_names[k])
    else:
        images[img_key] = reader.gdal_to_nparr(img_names[img_key])
    return images

def plot_whole_image(imageId):

    '''
    Description:
    The entire image is broken into a 4x4 grid.
    Hence there are 16 sub-squares(tiles) of the image.
    
    _0_0 ==> IMPLIES (0,0)
    _0_1 ==> IMPLIES (0,1)
    _0_2 ==> IMPLIES (0,2)
    _0_3 ==> IMPLIES (0,3) ans so on ...
    
    The above are top_most-left_most grid co-ordinates of the tile images
    This function takes the 16 images. Stitches them together and plots the 
    entire image.
    
    Arguments:
        imageId -- str
            imageId as used in grid_size.csv after removing the post-fix _x_y from the name
            ex. 6120 is correct and but 6120_x_y isn't
    
    '''    
    actual_str = imageId
    
        
    strA = actual_str +  '_0_0'
    strB = actual_str +  '_0_1'
    strC = actual_str +  '_0_2'
    strD = actual_str +  '_0_3'
    
    imageA = get_images(strA,'3');
    imageB = get_images(strB,'3');
    imageC = get_images(strC,'3');
    imageD = get_images(strD,'3');

    imageABCD = np.concatenate((imageA['3'],imageB['3'],imageC['3'],imageD['3']),axis = 2)
    
    imageA = None
    imageB = None
    imageC = None
    imageD = None

    strE = actual_str +  '_1_0'
    strF = actual_str +  '_1_1'
    strG = actual_str +  '_1_2'
    strH = actual_str +  '_1_3'
    
    imageE = get_images(strE,'3');
    imageF = get_images(strF,'3');
    imageG = get_images(strG,'3');
    imageH = get_images(strH,'3');
    imageEFGH = np.concatenate((imageE['3'],imageF['3'],imageG['3'],imageH['3']),axis = 2)

    imageE = None
    imageF = None
    imageG = None
    imageH = None
        
    strI = actual_str +  '_2_0'
    strJ = actual_str +  '_2_1'
    strK = actual_str +  '_2_2'
    strL = actual_str +  '_2_3'
    
    imageI = get_images(strI,'3');
    imageJ = get_images(strJ,'3');
    imageK = get_images(strK,'3');
    imageL = get_images(strL,'3');
    imageIJKL = np.concatenate((imageI['3'],imageJ['3'],imageK['3'],imageL['3']),axis = 2)
    
    imageI = None
    imageJ = None
    imageK = None
    imageL = None    

    strM = actual_str +  '_3_0'
    strN = actual_str +  '_3_1'
    strO = actual_str +  '_3_2'
    strP = actual_str +  '_3_3'
    
    imageM = get_images(strM,'3');
    imageN = get_images(strN,'3');
    imageO = get_images(strO,'3');
    imageP = get_images(strP,'3');
    imageMNOP = np.concatenate((imageM['3'],imageN['3'],imageO['3'],imageP['3']),axis = 2)
    
    imageM = None
    imageN = None
    imageO = None
    imageP = None

    strQ = actual_str +  '_4_0'
    strR = actual_str +  '_4_1'
    strS = actual_str +  '_4_2'
    strT = actual_str +  '_4_3'
    
    imageQ = get_images(strQ,'3');
    imageR = get_images(strR,'3');
    imageS = get_images(strS,'3');
    imageT = get_images(strT,'3');    
    imageQRST = np.concatenate((imageQ['3'],imageR['3'],imageS['3'],imageT['3']),axis = 2)
    
    imageQ = None
    imageR = None
    imageS = None
    imageT = None

    imageABCDEFGHIJKLMNOPQRST = np.concatenate((np.concatenate((imageABCD,imageEFGH),axis = 1) ,np.concatenate((imageIJKL,imageMNOP),axis = 1),imageQRST),axis = 1)

    imageABCD = None
    imageEFGH = None
    imageIJKL = None
    imageMNOP = None
    imageQRST = None
    
    plt.imshow(imageABCDEFGHIJKLMNOPQRST)

def get_size(imageId,rgb,channel = None):
    """
    Get the grid size of the image

    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    """
    
    xmax, ymin = gs[gs.ImageId == imageId].iloc[0,1:].astype(float)
    W, H = get_images(imageId, '3')['3'].shape[1:]
    return (xmax, ymin, W, H)


def is_training_image(imageId):
    '''
    Returns
    -------
    is_training_image : bool
        True if imageId belongs to training data
    '''
    
    if imageId in trainImageIds:
        return True
    
    return False


def plot_polygons(fig, ax, polygonsList):
    '''
    Plot descrates.PolygonPatch from list of polygons objs for each CLASS
    '''
    legend_patches = []
    for cType in polygonsList:
        print('{} : {} \tcount = {}'.format(cType, CLASSES[cType], len(polygonsList[cType])))
        legend_patches.append(Patch(color=COLORS[cType],
                                    label='{} ({})'.format(CLASSES[cType], len(polygonsList[cType]))))
        for polygon in polygonsList[cType]:
            mpl_poly = PolygonPatch(polygon,
                                    color=COLORS[cType],
                                    lw=0,
                                    alpha=0.7,
                                    zorder=ZORDER[cType])
            ax.add_patch(mpl_poly)
    # ax.relim()
    ax.autoscale_view()
    ax.set_title('Objects')
    ax.set_xticks([])
    ax.set_yticks([])
    return legend_patches


def plot_image(fig, ax, imageId, img_key, selected_channels=None):
    '''
    Plot get_images(imageId)[image_key] on axis/fig
    Optional: select which channels of the image are used (used for sixteen_band/ images)
    Parameters
    ----------
    img_key : str, {'3', 'P', 'N', 'A'}
        See get_images for description.
    '''
    images = get_images(imageId, img_key)
    img = images[img_key]
    title_suffix = ''
    if selected_channels is not None:
        img = img[selected_channels]
        title_suffix = ' (' + ','.join([ str(i) for i in selected_channels ]) + ')'
    if len(img.shape) == 2:
        new_img = np.zeros((3, img.shape[0], img.shape[1]))
        new_img[0] = img
        new_img[1] = img
        new_img[2] = img
        img = new_img
    
    plt.imshow(img, figure=fig, subplot=ax)
    ax.set_title(imageId + ' - ' + img_key + title_suffix)
    ax.set_xlabel(img.shape[-2])
    ax.set_ylabel(img.shape[-1])
    ax.set_xticks([])
    ax.set_yticks([])



def visualize_image(imageId,rgb = True,plot_all=True):
    '''         
    Plot all images and object-polygons
    
    Parameters
    ----------
    imageId : str
        imageId as used in grid_size.csv
    plot_all : bool, True by default
        If True, plots all images (from three_band/ and sixteen_band/) as subplots.
        Otherwise, only plots Polygons.
    '''         
    df_image = df[df.ImageId == imageId]
    
    xmax, ymin, W, H = get_size(imageId,rgb)
    
    
    if plot_all:
        fig, axArr = plt.subplots(figsize=(10, 10), nrows=3, ncols=3)
        ax = axArr[0][0]
    else:
        fig, axArr = plt.subplots(figsize=(10, 10))
        ax = axArr
    
    if is_training_image(imageId):
        
        # If the image is not a training image, then leave the first sub-plot empty
        # else fill the first subplot with the satellite image filled with colours
        
        print('ImageId : {}'.format(imageId))
        polygonsList = {}
        for cType in CLASSES.keys():
            polygonsList[cType] = wkt_loads(df_image[df_image.ClassType == cType].MultipolygonWKT.values[0])
        legend_patches = plot_polygons(fig, ax, polygonsList)
        
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, 0)
        ax.set_xlabel(xmax)
        ax.set_ylabel(ymin)
                
    if plot_all:
        
        # Plot 9 different photos in the remaining 9 sub-plots
        
        plot_image(fig, axArr[0][1], imageId, '3')
        plot_image(fig, axArr[0][2], imageId, 'P')
        plot_image(fig, axArr[1][0], imageId, 'A', [0, 3, 6]) # superimpose 3 images
        plot_image(fig, axArr[1][1], imageId, 'A', [1, 4, 7]) # superimpose 3 images
        plot_image(fig, axArr[1][2], imageId, 'A', [2, 5, 0]) # superimpose 3 images
        plot_image(fig, axArr[2][0], imageId, 'M', [0, 3, 6]) # superimpose 3 images
        plot_image(fig, axArr[2][1], imageId, 'M', [1, 4, 7]) # superimpose 3 images
        plot_image(fig, axArr[2][2], imageId, 'M', [2, 5, 0]) # superimpose 3 images

    if is_training_image(imageId):
        ax.legend(handles=legend_patches,
                   # loc='upper center',
                   bbox_to_anchor=(0.9, 1),
                   bbox_transform=plt.gcf().transFigure,
                   ncol=5,
                   fontsize='x-small',
                   title='Objects-' + imageId,
                   # mode="expand",
                   framealpha=0.3)
    return (fig, axArr, ax)


if __name__ == '__main__' :

    # test function
    #plot_whole_image('6120')

    # Current Directory
    inDir = os.getcwd()

    # read the training data from train_wkt_v4.csv
    df = pd.read_csv(os.path.join(inDir,'Data/train_wkt_v4.csv'))
    print(df.head())

    # grid size will also be needed later..
    gs = pd.read_csv(os.path.join(inDir,'Data/grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    print(gs.head())

    #  Distinct imageId NUMBERS in the dataframe
    distinct_train_id_nos = (get_distinct_id_nos(df))
    distinct_grid_id_nos = (get_distinct_id_nos(gs))

    # Distinct imageIds in the DataFrame
    allImageIds = get_distinct_ids(gs)
    trainImageIds = get_distinct_ids(df)


    # Loop over few training images and save to files
    for imageId in trainImageIds:
        print(imageId)
        fig, axArr, ax = visualize_image(imageId,plot_all = False)
        plt.savefig('./Pictures/Objects--' + imageId + '.png')
        plt.show()
        plt.clf()








    
    
    
    
