import os
from gdal_utilities import gdal_utils
import pandas as pd

def main():
    ''' Performs Pansharpening with M bands and Panchromatic Band.'''

    # Base Address
    inDir = os.getcwd()
    
    # Train-images multiploygon co-ordinates
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
    
    for imageID in trainImageIds:
        
        string_base = "gdal_sharpen.py "
                
        bands = "-b 1 -b 2 -b 3 -b 4 -b 5 -b 6 -b 7 -b 8 "
    
        panchro = os.path.join(inDir,"Data/sixteen_band/" + imageID + "_P.tif "  )
        print(panchro)

        m_band = os.path.join(inDir,"Data/sixteen_band/" + imageID + "_M.tif "  )        
        print(m_band)
        
        output = os.path.join(inDir,"Data/pan_sharpened_images/" + imageID + "_M.tif"  )
        print(output)
        
        final = string_base + bands + panchro + m_band + output
        print(final)
        
        creator.gdal_pansharpen(final)

if __name__ == '__main__':

    main()
