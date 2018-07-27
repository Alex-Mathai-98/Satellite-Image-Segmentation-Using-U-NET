#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:29:36 2018

@author: alex
"""

from osgeo import gdal
import osr
import numpy as np
import os
import sys

class gdal_utils :
    
    def __init__ (self):
        pass        
            
    # Convert gdal object to numpy array
    def gdal_to_nparr(self,path):
        
        """
        Description : Takes the path name to a tiff file and gets the numpy array of that raster.
        """
        
        # open dataset
        ds = gdal.Open(path) 
        
        if ds is None:
            print("The path provided is invalid")
            return None
        
        if ds.RasterCount <= 0:
            print("Gdal Object is Empty. Returning None.")
            return None
        else:
            # Cartesion like convention not normal programming convention
            cols = ds.RasterXSize
            rows = ds.RasterYSize
        
        arr = np.zeros(shape = (rows,cols,ds.RasterCount))
        
        for band in range(ds.RasterCount):
            arr[:,:,band] = np.array(ds.GetRasterBand(band + 1).ReadAsArray())
        
        return arr
    
    # Creating a tiff file from a numpy array
    def create_tiff_file_from_array(self,ref_raster_fn,new_raster_fn,array):
        '''
        Description :
            The Reference Raster must have the same dimensions as the array. 
            Saves a new raster file taking the "number of cols","number of rows","originX","originY","pixelWidth","pixelHeight","spatial reference"
            from the Reference Raster and the values from the array (Note that the values are rounded to the nearest integer in the raster) . 
            
        Parameters : 
            ref_raster_fn -- 	str.
            					Reference Raster Filename.
            new_raster_fn -- 	str.
            					New Raster Filename.
            array         --    np.array.
            					The array that contains the information.

		Returns :
			new_raster   --     np.array
                                A copy of the new tiff file that was saved.
        '''

        ref_raster = gdal.Open(ref_raster_fn)
        
        if ref_raster == None :
            raise ValueError("The reference raster file name is invalid.")
            
        if ref_raster.RasterCount <= 0:
            raise ValueError("Ref_Raster is an invalid raster. It has no bands")
        
        geotransform = ref_raster.GetGeoTransform()
        
        cols = ref_raster.RasterXSize
        rows = ref_raster.RasterYSize
        
        originX = geotransform[0]
        originY = geotransform[3]
        
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
    
        datatype = ref_raster.GetRasterBand(1).DataType 
    
        driver = gdal.GetDriverByName('GTiff')
        
        print("Rows : {}, Columns : {}, Channels : {} ".format(rows,cols,array.shape[2]))    
        
        outRaster = driver.Create(new_raster_fn,cols,rows,array.shape[2],datatype)
        outRaster.SetGeoTransform((originX,pixelWidth,0,originY,0,pixelHeight))
        
        # Copy all the channels
        for i in range(array.shape[2]):
            band = outRaster.GetRasterBand( i + 1 )
            band.WriteArray(array[:,:,i])
            
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(ref_raster.GetProjectionRef())
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        band.FlushCache()
        
        new_raster = gdal.Open(new_raster_fn)
        
        if new_raster is None:
            print("Array to Raster Conversion Failed")
        
        return new_raster
    
    # Helper function for self.gdal_pansharpen() if the arguments provided by the user are incorrect.
    def Usage(self):
        print('Usage: gdal_pansharpen [--help-general] pan_dataset {spectral_dataset[,band=num]}+ out_dataset')
        print('                       [-of format] [-b band]* [-w weight]*')
        print('                       [-r {nearest,bilinear,cubic,cubicspline,lanczos,average}]')
        print('                       [-threads {ALL_CPUS|number}] [-bitdepth val] [-nodata val]')
        print('                       [-spat_adjust {union,intersection,none,nonewithoutwarning}]')
        print('                       [-verbose_vrt] [-co NAME=VALUE]* [-q]')
        print('')
        print('Create a dataset resulting from a pansharpening operation.')
        return -1

    def gdal_pansharpen(self,input_string):

        ''' Modified version of gdal_pansharpen.py script 

            Arguments :
            input_string	--	str. 
                                Parameters as defined in the self.Usage() function defined above.

            Returns :
                1	-- If the algorithm failed
                0	-- If the algorithm succeeded (i.e. a new pansharpened image was created)
        '''

        argv = input_string.split(' ')
        
        if argv is None:
            return -1
    
        pan_name = None
        last_name = None
        spectral_ds = []
        spectral_bands = []
        out_name = None
        bands = []
        weights = []
        format = 'GTiff'
        creation_options = []
        callback = gdal.TermProgress
        resampling = None
        spat_adjust = None
        verbose_vrt = False
        num_threads = None
        bitdepth = None
        nodata = None
    
        i = 1
        argc = len(argv)
        while i < argc:
            if argv[i] == '-of' and i < len(argv)-1:
                format = argv[i+1]
                i = i + 1
            elif argv[i] == '-r' and i < len(argv)-1:
                resampling = argv[i+1]
                i = i + 1
            elif argv[i] == '-spat_adjust' and i < len(argv)-1:
                spat_adjust = argv[i+1]
                i = i + 1
            elif argv[i] == '-b' and i < len(argv)-1:
                bands.append(int(argv[i+1]))
                i = i + 1
            elif argv[i] == '-w' and i < len(argv)-1:
                weights.append(float(argv[i+1]))
                i = i + 1
            elif argv[i] == '-co' and i < len(argv)-1:
                creation_options.append(argv[i+1])
                i = i + 1
            elif argv[i] == '-threads' and i < len(argv)-1:
                num_threads = argv[i+1]
                i = i + 1
            elif argv[i] == '-bitdepth' and i < len(argv)-1:
                bitdepth = argv[i+1]
                i = i + 1
            elif argv[i] == '-nodata' and i < len(argv)-1:
                nodata = argv[i+1]
                i = i + 1
            elif argv[i] == '-q':
                callback = None
            elif argv[i] == '-verbose_vrt':
                verbose_vrt = True
            elif argv[i][0] == '-':
                sys.stderr.write('Unrecognized option : %s\n' % argv[i])
                return self.Usage()
            elif pan_name is None:
                pan_name = argv[i]
                pan_ds = gdal.Open(pan_name)
                if pan_ds is None:
                    return 1
            else:
                if last_name is not None:
                    pos = last_name.find(',band=')
                    if pos > 0:
                        spectral_name = last_name[0:pos]
                        ds = gdal.Open(spectral_name)
                        if ds is None:
                            return 1
                        band_num = int(last_name[pos+len(',band='):])
                        band = ds.GetRasterBand(band_num)
                        spectral_ds.append(ds)
                        spectral_bands.append(band)
                    else:
                        spectral_name = last_name
                        ds = gdal.Open(spectral_name)
                        if ds is None:
                            return 1
                        for j in range(ds.RasterCount):
                            spectral_ds.append(ds)
                            spectral_bands.append(ds.GetRasterBand(j+1))
    
                last_name = argv[i]
    
            i = i + 1
            
        print("Format : {}, Bands : {}, pan_fn : {}, spectral_fn : {}, output_fn : {} ".format(format,bands,pan_name,spectral_name,last_name) )
    
    
        if pan_name is None or len(spectral_bands) == 0:
            return self.Usage()
        out_name = last_name
        
        if len(bands) == 0:
            bands = [ j+1 for j in range(len(spectral_bands)) ]
        else:
            for i in range(len(bands)):
                if bands[i] < 0 or bands[i] > len(spectral_bands):
                    print('Invalid band number in -b: %d' % bands[i])
                    return 1
    
        if len(weights) != 0 and len(weights) != len(spectral_bands):
            print('There must be as many -w values specified as input spectral bands')
            return 1
    
        vrt_xml = """<VRTDataset subClass="VRTPansharpenedDataset">\n"""
        if bands != [ j+1 for j in range(len(spectral_bands)) ]:
            for i in range(len(bands)):
                band = spectral_bands[bands[i]-1]
                datatype = gdal.GetDataTypeName(band.DataType)
                colorname = gdal.GetColorInterpretationName(band.GetColorInterpretation())
                vrt_xml += """  <VRTRasterBand dataType="%s" band="%d" subClass="VRTPansharpenedRasterBand">
          <ColorInterp>%s</ColorInterp>
      </VRTRasterBand>\n""" % (datatype, i+1, colorname)
    
        vrt_xml += """  <PansharpeningOptions>\n"""
    
        if len(weights) != 0:
            vrt_xml += """      <AlgorithmOptions>\n"""
            vrt_xml += """        <Weights>"""
            for i in range(len(weights)):
                if i > 0: vrt_xml += ","
                vrt_xml += "%.16g" % weights[i]
            vrt_xml += "</Weights>\n"
            vrt_xml += """      </AlgorithmOptions>\n"""
    
        if resampling is not None:
            vrt_xml += '      <Resampling>%s</Resampling>\n' % resampling
    
        if num_threads is not None:
            vrt_xml += '      <NumThreads>%s</NumThreads>\n' % num_threads
    
        if bitdepth is not None:
            vrt_xml += '      <BitDepth>%s</BitDepth>\n' % bitdepth
    
        if nodata is not None:
            vrt_xml += '      <NoData>%s</NoData>\n' % nodata
    
        if spat_adjust is not None:
            vrt_xml += '      <SpatialExtentAdjustment>%s</SpatialExtentAdjustment>\n' % spat_adjust
    
        pan_relative='0'
        if format.upper() == 'VRT':
            if not os.path.isabs(pan_name):
                pan_relative='1'
                pan_name = os.path.relpath(pan_name, os.path.dirname(out_name))
    
        vrt_xml += """    <PanchroBand>
          <SourceFilename relativeToVRT="%s">%s</SourceFilename>
          <SourceBand>1</SourceBand>
        </PanchroBand>\n""" % (pan_relative, pan_name)
    
        for i in range(len(spectral_bands)):
            dstband = ''
            for j in range(len(bands)):
                if i + 1 == bands[j]:
                    dstband = ' dstBand="%d"' % (j+1)
                    break
    
            ms_relative='0'
            ms_name = spectral_ds[i].GetDescription()
            if format.upper() == 'VRT':
                if not os.path.isabs(ms_name):
                    ms_relative='1'
                    ms_name = os.path.relpath(ms_name, os.path.dirname(out_name))
    
            vrt_xml += """    <SpectralBand%s>
          <SourceFilename relativeToVRT="%s">%s</SourceFilename>
          <SourceBand>%d</SourceBand>
        </SpectralBand>\n""" % (dstband, ms_relative, ms_name, spectral_bands[i].GetBand())
    
        vrt_xml += """  </PansharpeningOptions>\n"""
        vrt_xml += """</VRTDataset>\n"""
    
        if format.upper() == 'VRT':
            f = gdal.VSIFOpenL(out_name, 'wb')
            if f is None:
                print('Cannot create %s' % out_name)
                return 1
            gdal.VSIFWriteL(vrt_xml, 1, len(vrt_xml), f)
            gdal.VSIFCloseL(f)
            if verbose_vrt:
                vrt_ds = gdal.Open(out_name, gdal.GA_Update)
                vrt_ds.SetMetadata(vrt_ds.GetMetadata())
            else:
                vrt_ds = gdal.Open(out_name)
            if vrt_ds is None:
                return 1
    
            return 0
    
        vrt_ds = gdal.Open(vrt_xml)
        out_ds = gdal.GetDriverByName(format).CreateCopy(out_name, vrt_ds, 0, creation_options, callback = callback)
        if out_ds is None:
            return 1
        return 0
