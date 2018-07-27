#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:18:33 2018

@author: alex
"""

import numpy as np
import cv2
import os 
import pandas as pd
from shapely.wkt import loads as wkt_loads
from gdal_utilities import gdal_utils

### Generate the masks for ground truth ###
    
class mask_generator :
    
    def __init__(self):
        
        self.CLASSES =  {
                            1 : 'Bldg',
                            2 : 'Struct',
                            3 : 'Road',
                            4 : 'Track',
                            5 : 'Trees',
                            6 : 'Crops',
                            7 : 'Fast_H20',
                            8 : 'Slow_H20',
                            9 : 'Truck',
                            10 : 'Car',
                        }
        
         

    def _convert_coordinates_to_raster(self,coords, img_size, xymax):
        """ Do the transformtions to the co-ordinates of the image - using the formula given.

        Arguments :
            coords --
            img_size --
            xymax --

        Returns :
            coords_int -- 

        """
        Xmax,Ymax = xymax
        H,W = img_size[0:2]
                
        W1 = 1.0*W*W/(W+1)
        H1 = 1.0*H*H/(H+1)
        xf = W1/Xmax
        yf = H1/Ymax
        coords[:,1] *= yf
        coords[:,0] *= xf
        coords_int = np.round(coords).astype(np.int32)
        return coords_int

    def _get_xmax_ymin(self,grid_sizes_panda, imageId):
        """ Returns the xmax and ymin of the photographs.

        Arguments : 
            grid_sizes_panda -- pandas.DataFrame Object
                                The pandas dataframe with all the grid sizes for each image

            imageId          -- str.
                                The id of the image. ex. 6010_0_0.tif

        Returns :
            (xmax,ymin) -- tuple.
                           The maximum x co-ordinate and the minimum y co-ordinate
        """
        xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
        return (xmax,ymin)


    def _get_polygon_list(self,wkt_list_pandas, imageId, cType):
        """ Gets the list of polygons that were created for class : "cType" in image with id : "imageId"

        Arguments :
            wkt_list_pandas -- pandas.DataFrame Object
                               The pndas DataFrame with all the shape files of the satellite image.
            imageId         -- str.
                               The id of the image. ex. 6010_0_0.tif
            cType           -- str.
                               The name of the class type. ex 'Bldg','Struct'
        Returns :
            polygonList -- 

        """
        df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
        multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
        polygonList = None
        if len(multipoly_def) > 0:
            assert len(multipoly_def) == 1
            polygonList = wkt_loads(multipoly_def.values[0])
        return polygonList


    def _get_and_convert_contours(self,polygonList, raster_img_size, xymax):
        """
        Converts the co-ordindates of the polygons using the transformation rules that were stated.
        It then returns two sets of co-ordinates - the outer contour of the polygons "perim-list", and the inner contour of the polygons "interior_list"
        """
        perim_list = []
        interior_list = []
        if polygonList is None:
            return None
        
        for k in range(len(polygonList)):
                        
            # Get the outer contours of the polygons and add to the perim_list
            poly = polygonList[k]
            perim = np.array(list(poly.exterior.coords))
            perim_c = self._convert_coordinates_to_raster(perim, raster_img_size, xymax)
            perim_list.append(perim_c)
            
            # For each polygon get the interior contours of the polygons and add to the interior_list
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = self._convert_coordinates_to_raster(interior, raster_img_size, xymax)
                interior_list.append(interior_c)
                
        return perim_list,interior_list


    def _plot_mask_from_contours(self,raster_img_size, contours, class_value = 1):
        """
        Given the inner contour of the polygons and the outer contours of the polygons now plot the mask of polygons with 1 between
        the region of the inner and outer contours and 0 otherwise.
        
        Hence this gives us a mask that highlights only the regions of interest
        """
        
        img_mask = np.zeros(raster_img_size,np.uint8)
        
        if contours is None:
            return img_mask
        
        # perim_list and interior_list
        perim_list,interior_list = contours
        
        # fill 1 inside the boundaries of the perim list
        cv2.fillPoly(img_mask,perim_list,class_value)
        
        # fill 0 inside the boundaries of the interior list
        cv2.fillPoly(img_mask,interior_list,0)
        
        return img_mask


    def generate_mask_for_image_and_class(self,raster_size, imageId, class_type, grid_sizes_panda,wkt_list_pandas):
        """
        Generates the complete mask of the image given the "class_type" and "imageId".
        """
        xymax = self._get_xmax_ymin(grid_sizes_panda,imageId)
        
        polygon_list = self._get_polygon_list(wkt_list_pandas,imageId,class_type)
        
        contours = self._get_and_convert_contours(polygon_list,raster_size,xymax)
        
        mask = self._plot_mask_from_contours(raster_size,contours,1)
        
        return mask


    def get_distinct_ids(self,df):
        ''' Returns a list of distinct Ids which are sorted from the pandas dataframe. '''
        ans = []
        
        for i in range(len(df)):
            string = (df.iloc[i,0])
            
            if string not in ans:
               ans.append(string) 
               
            
        print("The are " + str(len(ans)) + " distinct ids." + str(df.head()))
        
        ans.sort()
         
        return ans
    
    def get_image_names(self,imageId):
        '''
        Get the names of the tiff files
        '''
        
        inDir = os.getcwd()
        
        d = {'3': '{}/Data/pan_sharpened_images/{}.tif'.format(inDir, imageId),
             'A': '{}/Data/sixteen_band/{}_A.tif'.format(inDir, imageId),
             'M': '{}/Data/sixteen_band/{}_M.tif'.format(inDir, imageId),
             'P': '{}/Data/sixteen_band/{}_P.tif'.format(inDir, imageId),
             }
        return d


    def get_images(self,imageId, img_key = None):
        '''
        Load images correspoding to imageId
    
        Parameters
        ----------
        imageId : str
            imageId as used in grid_size.csv
        img_key : str 
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
        
        img_names = self.get_image_names(imageId)
        images = dict()
        if img_key is None:
            for k in img_names.keys():
                images[k] = creator.gdal_to_nparr(img_names[k])
        else:
            images[img_key] = creator.gdal_to_nparr(img_names[img_key])
        return images

    def generate_all_masks(self):
        
        # Base Address
        inDir = os.getcwd()
        
        # Helper class
        creator = gdal_utils()
        
        # train-images multiploygon co-ordinates
        df = pd.read_csv(os.path.join(inDir,'Data/train_wkt_v4.csv'))
        print(df.head())
    
        # grid size will also be needed later..
        gs = pd.read_csv(os.path.join(inDir,'Data/grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
        print(gs.head())
        
        # Distinct imageIds in the DataFrame
        trainImageIds = self.get_distinct_ids(df)    
        
        for key,classes in enumerate(self.CLASSES) :        
            
            base = os.path.join(os.getcwd(),'Data_masks')        
            path = os.path.join(base,self.CLASSES[classes])
            
            print(base)
            print(path)
            print(classes)
            
            for imageId in trainImageIds :
                                
                image = self.get_images(imageId,'P')
                print("imageId : {}, image_shape : {}".format(imageId,image['P'].shape) )
                                
                mask = self.generate_mask_for_image_and_class(image['P'].shape,imageId,classes,gs,df)
                print("mask : {}".format(mask.shape))
                
                ref_raster_fn = os.path.join(os.getcwd(),"Data/sixteen_band/" + imageId + "_P.tif")
                new_raster_fn = os.path.join(path,imageId + "_" + self.CLASSES[classes] + ".tif")
                
                temp = creator.create_tiff_file_from_array(ref_raster_fn,new_raster_fn,mask*255)
                temp = None
                
if __name__ == '__main__' :
    
    alex = mask_generator()
    alex.generate_all_masks()




