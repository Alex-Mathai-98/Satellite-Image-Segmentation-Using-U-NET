from gdal_utilities import gdal_utils
import numpy as np
import os
from sklearn.externals import joblib
import pickle
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.ndimage import binary_opening,binary_closing

def get_masks_list():
    
    '''
    Gets the list of masked tiff files intended for training and testing 
    '''
    
    inDir = os.getcwd()    
    
    files_train_temp = os.listdir(os.path.join(inDir,"Data_masks/Fast_H20"))
    files_train_final = []
    
    i = 0 
    for file in files_train_temp : 
        
        extension = os.path.splitext(file) 
    
        if extension[1] == '.tif' :
            files_train_final.append(file[:-13])
    
        i += 1
    
    
    files_test_temp = os.listdir( os.path.join(inDir,"Data_masks/Fast_H20/Test") )
    files_test_final = []
    
    for file in files_test_temp:
        files_test_final.append(file[:-13])
    
    return files_train_final,files_test_final


def get_testing_image_pair(files_test,imageId):
    ''' Gets the input,truth for an image.

    Description :
        Get image-ground-truth pair of the image with id "imageId" from the list of
        testing images "files_test"

    Arguments :
        files_test  -- List.
                       The list of names of the tiff images that will be used for testing
        imageId     -- String.
                       The id of the image.
                       ex. '6010_0_0'
    Returns :
        (image_train,truth) -- Tuple.
                               This tuple containing the input image and the ground truth.
    '''
    if imageId not in files_test:
        raise ValueError("Invalid value of imageId")
        return None

    # Using gdal to read the input image
    reader = gdal_utils()
    path = os.path.join(os.getcwd(),"Data/image_stacks/" + imageId + ".tif")
    image_test = reader.gdal_to_nparr(path)
    
    if image_test is None:
        print("Failed to load image. Wrong path name provided.")
        return None
    
    path = os.path.join(os.getcwd(),"Data_masks/Fast_H20/Test/" + imageId + "_Fast_H20.tif")

    # Using gdal to read the ground truth    
    truth = reader.gdal_to_nparr(path)
    if truth is None:
        print("Failed to load ground truth. Wrong path name provided.")
        return None
    
    return (image_test,truth)

def get_classifiers():
    ''' Gets the trained classfiers for classification.'''

    path = os.getcwd()
    
    with open( os.path.join(path,'Parameters/Fast_H20/sgd_svm.pkl') ,'rb') as f :
        sgd_svm = pickle.load(f)

    with open( os.path.join(path,'Parameters/Fast_H20/sgd_lr.pkl') ,'rb') as f :
        sgd_lr = pickle.load(f)

    joblib_file = os.path.join(path,'Parameters/Fast_H20/rfc.pkl')
    rfc = joblib.load(joblib_file)


    return sgd_svm,sgd_lr,rfc


def get_std_scaler():
    ''' Get the saved Standard_Scaler Object ("std_scaler").'''

    path = os.getcwd()

    with open( os.path.join(path,'Parameters/Fast_H20/std_scaler.pkl') ,'rb') as f :
        std_scaler = pickle.load(f)
    
    return std_scaler

def normalizer(image,std_scaler) :
    ''' Normalizes the input "image" with train-data mean and std-dev.

    Arguments :
        image      -- np.array.
                      The array that needs to be normalized.
        std_scaler -- sklearn.preprocessing.StandardScaler Object.
                      The standard scaler that's already fit on the
                      training data.

    Returns :
        x - The normalized form of the input. 
    
    '''
    x = np.reshape(image , newshape = (-1,14))
    x = std_scaler.transform(x)
    return x

def predict(image,truth,sgd_svm,sgd_lr,rfc):
    ''' Predicts output on the test data.

    Description :
        Max voting is done on the prediction from the three machine learning
        models and the final output for each pixel is calculated.

        Max voting is the same as rounding the averaged results.

    Arguments :

        image         -- np.array.
                         The image whose ground truth needs to be predicted.
        truth         -- np.array.
                         The ground truth of the input "image"
        sgd_svm       -- sklearn.linear_model.SGDClassifier Object.
                         A trained support vector machine.
        sgd_lr        -- sklearn.linear_model.SGDClassifier Object.
                         A trained logistic regression model.
        rfc           -- sklearn.ensemble.RandomForestClassifier Object.
                         A trained random forest classifer.

    Returns :
        avg_pred -  np.array
                    The final result after max_voting.
    '''

    # Make Predictions
    pred_svm = sgd_svm.predict(image)
    pred_lr = sgd_lr.predict(image)
    pred_rfc = rfc.predict(image)

    # Reshaping
    pred_svm = np.reshape(pred_svm,newshape = (pred_svm.shape[0],1))
    pred_lr = np.reshape(pred_lr,newshape = (pred_lr.shape[0],1))	
    pred_rfc = np.reshape(pred_rfc,newshape = (pred_rfc.shape[0],1))

    # Max Voting
    avg_pred = np.round_((pred_svm + pred_lr + pred_rfc)/3)

    # Free Memory
    pred_rfc = None
    pred_lr  = None
    pred_svm = None

    # Reshaping outputs to match ground truth dimensions
    avg_pred = np.reshape(avg_pred,newshape = (truth.shape[0],truth.shape[1],1))
    
    return avg_pred

def jaccard_index(avg_pred,truth) :
    ''' Calculate the jaccard index.

    Arguments :
        avg_pred -- np.array.
                    The max voted predictions from the ensembled machine learning model.
        truth    -- np.array.
                    The ground truth of the input "avg_pred"

    Returns :
        Jaccard Index - (A intersection B)/(A union B)
    
    '''
    
    nr = np.sum(np.multiply(avg_pred,truth))

    dr = np.sum(avg_pred) + np.sum(truth)

    return (nr + 1e-12)/(dr - nr + 1e-12)


def get_results(image_test,truth) :
    '''Returns the prediction of "image_test" and calculates its jaccard index.

    Arguments :
        image_test -- np.array.
                      The image whose ground truth needs to be predicted.
        truth      -- np.array.
                      The ground truth of the input "image_test"
    Returns :
        avg_pred  - np.array.
                    The final prediction from the image_test
    '''

    sgd_svm,sgd_lr,rfc = get_classifiers()
    std_scaler = get_std_scaler()

    image_test = normalizer(image_test,std_scaler)

    avg_pred = predict(image_test,truth,sgd_svm,sgd_lr,rfc)

    metric = jaccard_index(avg_pred,truth/255)

    print("Jaccard Index : {}".format(metric))
    
    return avg_pred

def apply_median_filter(avg_pred):
    ''' Applies the median filter to the output to remove salt and pepper noise.'''
    
    avg_pred = np.reshape( avg_pred, newshape = (avg_pred.shape[0],avg_pred.shape[1]) )    
    avg_pred = medfilt2d(avg_pred)
    
    return avg_pred


def post_processing(avg_pred) : 
    ''' Applies the median filter and cleans the image noise using morphology operators.'''
    
    avg_pred = apply_median_filter(avg_pred)
    avg_pred = np.expand_dims(avg_pred,axis = 2)
    avg_pred = np.asarray(avg_pred,dtype = np.uint8)    

    # Structure for cleaning noise
    structure = np.ones( shape = (20,20,1) )

    # Morphology Opening and Closing 
    opened = binary_opening(avg_pred,structure)
    closed = binary_closing(opened,structure)
    
    return closed

if __name__ == '__main__' :

    files_train,files_test = get_masks_list()
    print(files_test)
    
    imageId = files_test[0]
    image_test,truth = get_testing_image_pair(files_test,imageId)
    
    avg_pred = get_results(image_test,truth)
    metric = jaccard_index(avg_pred,truth/255)
    print("Jaccard Index : {}".format(metric))
    
    writer_1 = gdal_utils()
    writer_1.create_tiff_file_from_array(os.path.join(os.getcwd(),"Data_masks/Fast_H20/Test/" + imageId + "_Fast_H20.tif"),os.path.join(os.getcwd(),"Results/pred_Fast_H20.tif"),avg_pred*255)
    
    closed = post_processing(avg_pred)
    metric = jaccard_index(closed,truth/255)
    print("Jaccard Index : {}".format(metric))
    
    plt.imshow(closed[:,:,0])
    
    writer_2 = gdal_utils()
    writer_2.create_tiff_file_from_array(os.path.join(os.getcwd(),"Data_masks/Fast_H20/Test/" + imageId + "_Fast_H20.tif"),os.path.join(os.getcwd(),"Results/post_process_pred_Fast_H20.tif"),closed*255)
    


    




