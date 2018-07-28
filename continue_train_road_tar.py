#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:07:28 2018

@author: alex
"""
from up_convolution import convolution,trans_convolve
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

############################################ NETWORK BUILDING ############################################
def get_bilinear_filter(filter_shape, upscale_factor):
    '''
    Description :  Generates a filter than performs simple bilinear interpolation for a given upsacle_factor
    
    Arguments:
        filter_shape -- [width, height, num_in_channels, num_out_channels] -> num_in_channels = num_out_channels
        upscale_factor -- The number of times you want to scale the image.
        
    Returns :
        weigths -- The populated bilinear filter
    '''
    
    kernel_size = filter_shape[1]
    
    # Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5
 
    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    
    for k in range(filter_shape[3]):
        for i in range(filter_shape[2]):
            weights[:, :, i, k] = bilinear
        
    return weights    

def variable_summaries_weights_biases(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    tf.summary.histogram('histogram',var)

def variable_summaries_scalars(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    tf.summary.scalar('value',var)

def create_placeholders(n_H0,n_W0,n_C0):
    """
    Creates the placeholders for the input size and for the number of output classes.
    
    Arguments:
    n_W0 -- scalar, width of an input matrix
    n_C0 -- scalar, number of channels of the input
    n_y  -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    
    with tf.name_scope("Inputs") :
        # Keep the number of examples as a variable (None) and the height of the matrix as variables (None)
        X = tf.placeholder(dtype = tf.float32, shape = (None,n_H0,n_W0,n_C0), name = "X") 
        Y = tf.placeholder(dtype = tf.float32, shape = (None,n_H0,n_W0,1), name = "Y")
    
    
    return X,Y


def initialize_parameters():
    '''
    Description:
        Initialize weight parameters for the weight matrix.

    Returns: 
        weight_parameters - A dictionary containing all the weights of the neural network
    '''
    
    left_1_1_conv = tf.get_variable(name = "Road_tar_left_1_1_conv",shape = (3,3,9,32),dtype = tf.float32,trainable = True)
    left_1_1_conv_bias = tf.get_variable(name = "Road_tar_left_1_1_conv_bias",shape = (32),dtype = tf.float32,trainable = True)
    
    left_1_2_conv = tf.get_variable(name = "Road_tar_left_1_2_conv",shape = (3,3,32,32),dtype = tf.float32,trainable = True)
    left_1_2_conv_bias = tf.get_variable(name = "Road_tar_left_1_2_conv_bias",shape = (32),dtype = tf.float32,trainable = True)

    left_2_1_conv = tf.get_variable(name = "Road_tar_left_2_1_conv",shape = (3,3,32,64),dtype = tf.float32,trainable = True)
    left_2_1_conv_bias = tf.get_variable(name = "Road_tar_left_2_1_conv_bias",shape = (64),dtype = tf.float32,trainable = True)
    
    left_2_2_conv = tf.get_variable(name = "Road_tar_left_2_2_conv",shape = (3,3,64,64),dtype = tf.float32,trainable = True)
    left_2_2_conv_bias = tf.get_variable(name = "Road_tar_left_2_2_conv_bias",shape = (64),dtype = tf.float32,trainable = True)    
    
    left_3_1_conv = tf.get_variable(name = "Road_tar_left_3_1_conv",shape = (3,3,64,128),dtype = tf.float32,trainable = True)
    left_3_1_conv_bias = tf.get_variable(name = "Road_tar_left_3_1_conv_bias",shape = (128),dtype = tf.float32,trainable = True)

    left_3_2_conv = tf.get_variable(name = "Road_tar_left_3_2_conv",shape = (3,3,128,128),dtype = tf.float32,trainable = True)
    left_3_2_conv_bias = tf.get_variable(name = "Road_tar_left_3_2_conv_bias",shape = (128),dtype = tf.float32,trainable = True)
    
    left_4_1_conv = tf.get_variable(name = "Road_tar_left_4_1_conv",shape = (3,3,128,256),dtype = tf.float32,trainable = True)
    left_4_1_conv_bias = tf.get_variable(name = "Road_tar_left_4_1_conv_bias",shape = (256),dtype = tf.float32,trainable = True)    
    
    left_4_2_conv = tf.get_variable(name = "Road_tar_left_4_2_conv",shape = (3,3,256,256),dtype = tf.float32,trainable = True)
    left_4_2_conv_bias = tf.get_variable(name = "Road_tar_left_4_2_conv_bias",shape = (256),dtype = tf.float32,trainable = True)        
    
    centre_5_1_conv = tf.get_variable(name = "Road_tar_centre_5_1_conv",shape = (3,3,256,512),dtype = tf.float32,trainable = True)
    centre_5_1_conv_bias = tf.get_variable(name = "Road_tar_centre_5_1_conv_bias",shape = (512),dtype = tf.float32,trainable = True)    
    
    centre_5_2_conv = tf.get_variable(name = "Road_tar_centre_5_2_conv",shape = (3,3,512,512),dtype = tf.float32,trainable = True)
    centre_5_2_conv_bias = tf.get_variable(name = "Road_tar_centre_5_2_conv_bias",shape = (512),dtype = tf.float32,trainable = True)

    centre_5_3_deconv = tf.get_variable(name = "Road_tar_centre_5_3_deconv",shape = (2,2,128,512),dtype = tf.float32,trainable = False)         

    right_4_1_conv = tf.get_variable(name = "Road_tar_right_4_1_conv",shape = (3,3,128 + 256,256),dtype = tf.float32,trainable = True)
    right_4_1_conv_bias = tf.get_variable(name = "Road_tar_right_4_1_conv_bias",shape = (256),dtype = tf.float32,trainable = True)
    
    right_4_2_conv = tf.get_variable(name = "Road_tar_right_4_2_conv",shape = (3,3,256,256),dtype = tf.float32,trainable = True)
    right_4_2_conv_bias = tf.get_variable(name = "Road_tar_right_4_2_conv_bias",shape = (256),dtype = tf.float32,trainable = True)

    right_4_3_deconv = tf.get_variable(name = "Road_tar_right_4_3_deconv",shape = (2,2,256,256),dtype = tf.float32,trainable = False)         
    
    right_3_1_conv = tf.get_variable(name = "Road_tar_right_3_1_conv",shape = (3,3,128 + 256,128),dtype = tf.float32,trainable = True)
    right_3_1_conv_bias = tf.get_variable(name = "Road_tar_right_3_1_conv_bias",shape = (128),dtype = tf.float32,trainable = True)
    
    right_3_2_conv = tf.get_variable(name = "Road_tar_right_3_2_conv",shape = (3,3,128,128),dtype = tf.float32,trainable = True)
    right_3_2_conv_bias = tf.get_variable(name = "Road_tar_right_3_2_conv_bias",shape = (128),dtype = tf.float32,trainable = True)

    right_3_3_deconv = tf.get_variable(name  = "Road_tar_right_3_3_deconv", shape = (2,2,128,128),dtype = tf.float32,trainable = False)

    right_2_1_conv = tf.get_variable(name = "Road_tar_right_2_1_conv",shape = (3,3,128 + 64,64),dtype = tf.float32,trainable = True)
    right_2_1_conv_bias = tf.get_variable(name = "Road_tar_right_2_1_conv_bias",shape = (64),dtype = tf.float32,trainable = True)
    
    right_2_2_conv = tf.get_variable(name = "Road_tar_right_2_2_conv",shape = (3,3,64,64),dtype = tf.float32,trainable = True)
    right_2_2_conv_bias = tf.get_variable(name = "Road_tar_right_2_2_conv_bias",shape = (64),dtype = tf.float32,trainable = True)

    right_2_3_deconv = tf.get_variable(name = "Road_tar_right_2_3_deconv",shape = (2,2,64,64),dtype = tf.float32,trainable = False)

    right_1_1_conv = tf.get_variable(name = "Road_tar_right_1_1_conv",shape = (9,9,64+32,32),dtype = tf.float32,trainable = True)
    right_1_1_conv_bias = tf.get_variable(name = "Road_tar_right_1_1_conv_bias",shape = (32),dtype = tf.float32,trainable = True)
    
    right_1_2_conv = tf.get_variable(name = "Road_tar_right_1_2_conv",shape = (9,9,32,1),dtype = tf.float32,trainable = True)
    right_1_2_conv_bias = tf.get_variable(name = "Road_tar_right_1_2_conv_bias",shape = (1),dtype = tf.float32,trainable = True)
    
    weight_parameters = {}

    weight_parameters["left_1_1_conv"] = left_1_1_conv
    weight_parameters["left_1_1_conv_bias"] = left_1_1_conv_bias
    
    weight_parameters["left_1_2_conv"] = left_1_2_conv
    weight_parameters["left_1_2_conv_bias"] = left_1_2_conv_bias

    weight_parameters["left_2_1_conv"] = left_2_1_conv
    weight_parameters["left_2_1_conv_bias"] = left_2_1_conv_bias    
    
    weight_parameters["left_2_2_conv"] = left_2_2_conv
    weight_parameters["left_2_2_conv_bias"] = left_2_2_conv_bias    

    weight_parameters["left_3_1_conv"] = left_3_1_conv
    weight_parameters["left_3_1_conv_bias"] = left_3_1_conv_bias        
    
    weight_parameters["left_3_2_conv"] = left_3_2_conv
    weight_parameters["left_3_2_conv_bias"] = left_3_2_conv_bias        

    weight_parameters["left_4_1_conv"] = left_4_1_conv
    weight_parameters["left_4_1_conv_bias"] = left_4_1_conv_bias            
    
    weight_parameters["left_4_2_conv"] = left_4_2_conv
    weight_parameters["left_4_2_conv_bias"] = left_4_2_conv_bias            
        
    weight_parameters["centre_5_1_conv"] = centre_5_1_conv
    weight_parameters["centre_5_1_conv_bias"] = centre_5_1_conv_bias                
    
    weight_parameters["centre_5_2_conv"] = centre_5_2_conv
    weight_parameters["centre_5_2_conv_bias"] = centre_5_2_conv_bias                

    weight_parameters["centre_5_3_deconv"] = centre_5_3_deconv

    weight_parameters["right_4_1_conv"] = right_4_1_conv
    weight_parameters["right_4_1_conv_bias"] = right_4_1_conv_bias            
    
    weight_parameters["right_4_2_conv"] = right_4_2_conv
    weight_parameters["right_4_2_conv_bias"] = right_4_2_conv_bias

    weight_parameters["right_4_3_deconv"] = right_4_3_deconv

    weight_parameters["right_3_1_conv"] = right_3_1_conv
    weight_parameters["right_3_1_conv_bias"] = right_3_1_conv_bias        
    
    weight_parameters["right_3_2_conv"] = right_3_2_conv
    weight_parameters["right_3_2_conv_bias"] = right_3_2_conv_bias
    
    weight_parameters["right_3_3_deconv"] = right_3_3_deconv
    
    weight_parameters["right_2_1_conv"] = right_2_1_conv
    weight_parameters["right_2_1_conv_bias"] = right_2_1_conv_bias
    
    weight_parameters["right_2_2_conv"] = right_2_2_conv
    weight_parameters["right_2_2_conv_bias"] = right_2_2_conv_bias    
    
    weight_parameters["right_2_3_deconv"] = right_2_3_deconv
     
    weight_parameters["right_1_1_conv"] = right_1_1_conv
    weight_parameters["right_1_1_conv_bias"] = right_1_1_conv_bias

    weight_parameters["right_1_2_conv"] = right_1_2_conv
    weight_parameters["right_1_2_conv_bias"] = right_1_2_conv_bias
     
    return weight_parameters


def forward_prop(X,weight_parameters,bool_train = True) : 
    
    '''
    Description :
        Performs the forward propagation in the network.
        
    Arguments :
        X                 -- np.array
                             The input matrix
        weight_parameters -- dict.
                             The initialized weights for the matrix
        bool_train        -- Bool.
                             An argument passed to the batch normalization parameter, to allow the updation of batch mean and variance

    Returns :
        conv18 -- The final feature vector
    '''
    
    left_1_1_conv = weight_parameters["left_1_1_conv"] 
    left_1_2_conv = weight_parameters["left_1_2_conv"]
    
    left_2_1_conv = weight_parameters["left_2_1_conv"]
    left_2_2_conv = weight_parameters["left_2_2_conv"]
    
    left_3_1_conv = weight_parameters["left_3_1_conv"]
    left_3_2_conv = weight_parameters["left_3_2_conv"]
    
    left_4_1_conv = weight_parameters["left_4_1_conv"]
    left_4_2_conv = weight_parameters["left_4_2_conv"]
    
    centre_5_1_conv = weight_parameters["centre_5_1_conv"]
    centre_5_2_conv = weight_parameters["centre_5_2_conv"]

    left_1_1_conv_bias = weight_parameters["left_1_1_conv_bias"] 
    left_1_2_conv_bias = weight_parameters["left_1_2_conv_bias"]
    
    left_2_1_conv_bias = weight_parameters["left_2_1_conv_bias"]
    left_2_2_conv_bias = weight_parameters["left_2_2_conv_bias"]
    
    left_3_1_conv_bias = weight_parameters["left_3_1_conv_bias"]
    left_3_2_conv_bias = weight_parameters["left_3_2_conv_bias"]
    
    left_4_1_conv_bias = weight_parameters["left_4_1_conv_bias"]
    left_4_2_conv_bias = weight_parameters["left_4_2_conv_bias"]
    
    centre_5_1_conv_bias = weight_parameters["centre_5_1_conv_bias"]
    centre_5_2_conv_bias = weight_parameters["centre_5_2_conv_bias"]

    centre_5_3_deconv = weight_parameters["centre_5_3_deconv"]

    right_4_1_conv = weight_parameters["right_4_1_conv"] 
    right_4_1_conv_bias = weight_parameters["right_4_1_conv_bias"]             
    
    right_4_2_conv = weight_parameters["right_4_2_conv"] 
    right_4_2_conv_bias = weight_parameters["right_4_2_conv_bias"] 

    right_4_3_deconv = weight_parameters["right_4_3_deconv"]

    right_3_1_conv = weight_parameters["right_3_1_conv"]
    right_3_1_conv_bias = weight_parameters["right_3_1_conv_bias"]         
    
    right_3_2_conv = weight_parameters["right_3_2_conv"] 
    right_3_2_conv_bias = weight_parameters["right_3_2_conv_bias"]
    
    right_3_3_deconv = weight_parameters["right_3_3_deconv"]
    
    right_2_1_conv = weight_parameters["right_2_1_conv"]
    right_2_1_conv_bias = weight_parameters["right_2_1_conv_bias"]
    
    right_2_2_conv = weight_parameters["right_2_2_conv"] 
    right_2_2_conv_bias = weight_parameters["right_2_2_conv_bias"]   
    
    right_2_3_deconv = weight_parameters["right_2_3_deconv"]
     
    right_1_1_conv = weight_parameters["right_1_1_conv"] 
    right_1_1_conv_bias = weight_parameters["right_1_1_conv_bias"] 

    right_1_2_conv = weight_parameters["right_1_2_conv"] 
    right_1_2_conv_bias = weight_parameters["right_1_2_conv_bias"] 


    ### Left Branch 1st Layer ###
    
    
    ## INTERESTING -- TENSORFLOW DOES A BAD JOB WHEN WE WANT TO PAD AN EVEN INPUT WITH AN ODD KERNEL ##    
    with tf.name_scope("Left_Branch_1st_Layer") :
        
        with tf.name_scope("Conv_1") :
            conv1 = tf.nn.conv2d(tf.pad(X,paddings = [[0,0],[112,112],[112,112],[0,0]],mode = 'SYMMETRIC'),left_1_1_conv,strides = (1,3,3,1),padding = "VALID",name = "convolve")
            conv1 = tf.nn.bias_add(conv1,left_1_1_conv_bias,name = "bias_add")
            conv1 = tf.layers.batch_normalization(conv1,training = bool_train,name = "norm")
            conv1 = tf.nn.leaky_relu (conv1,name = "activation")
            variable_summaries_weights_biases(left_1_1_conv)
            variable_summaries_weights_biases(left_1_1_conv_bias)
    
        with tf.name_scope("Conv_2") :    
            conv2 = tf.nn.conv2d(tf.pad(conv1,paddings = [[0,0],[112,112],[112,112],[0,0]],mode = 'SYMMETRIC'), left_1_2_conv, (1,3,3,1), padding = "VALID",name = "convolve")
            conv2 = tf.nn.bias_add(conv2,left_1_2_conv_bias,name = "bias_add")
            conv2 = tf.layers.batch_normalization(conv2,training = bool_train,name = "norm_2")
            conv2 =  tf.nn.leaky_relu(conv2,name = "activation")
            variable_summaries_weights_biases(left_1_2_conv)
            variable_summaries_weights_biases(left_1_2_conv_bias)
        
        with tf.name_scope("Pool") :
            max_pool_1 = tf.nn.max_pool(tf.pad(conv2,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),ksize = (1,2,2,1), strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    
    ### Left Branch 2nd layer ###
    
    with tf.name_scope("Left_Branch_2nd_Layer") :   

        with tf.name_scope("Conv_1") :
            conv3 = tf.nn.conv2d(tf.pad(max_pool_1,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),left_2_1_conv, (1,3,3,1), padding = "VALID",name = "convolve")
            conv3 = tf.nn.bias_add(conv3,left_2_1_conv_bias,name = "bias_add")
            conv3 = tf.layers.batch_normalization(conv3,training = bool_train,name = "norm_3")
            conv3 =  tf.nn.leaky_relu(conv3,name = "activation")
            variable_summaries_weights_biases(left_2_1_conv)
            variable_summaries_weights_biases(left_2_1_conv_bias)

        with tf.name_scope("Conv_2") :
            conv4 = tf.nn.conv2d(tf.pad(conv3,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),left_2_2_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv4 = tf.nn.bias_add(conv4,left_2_2_conv_bias,name = "bias_add")
            conv4 = tf.layers.batch_normalization(conv4,training = bool_train,name = "norm_4")
            conv4 =  tf.nn.leaky_relu(conv4,name = "activation")
            variable_summaries_weights_biases(left_2_2_conv)
            variable_summaries_weights_biases(left_2_2_conv_bias)

        with tf.name_scope("Pool") :
            max_pool_2 = tf.nn.max_pool(conv4,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")

    
    ### Left Branch 3rd layer ###
    
    with tf.name_scope("Left_Branch_3rd_Layer") :
    
        with tf.name_scope("Conv_1") :
            conv5 = tf.nn.conv2d(tf.pad(max_pool_2,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),left_3_1_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv5 = tf.nn.bias_add(conv5,left_3_1_conv_bias,name = "bias_add")
            conv5 = tf.layers.batch_normalization(conv5,training = bool_train,name = "norm_5")
            conv5 = tf.nn.leaky_relu(conv5,name = "activation")
            variable_summaries_weights_biases(left_3_1_conv)
            variable_summaries_weights_biases(left_3_1_conv_bias)

        with tf.name_scope("Conv_2") :
            conv6 = tf.nn.conv2d(tf.pad(conv5,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),left_3_2_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv6 = tf.nn.bias_add(conv6,left_3_2_conv_bias,name = "bias_add")
            conv6 = tf.layers.batch_normalization(conv6,training = bool_train,name = "norm_6")
            conv6 = tf.nn.leaky_relu(conv6,name = "activation")
            variable_summaries_weights_biases(left_3_2_conv)
            variable_summaries_weights_biases(left_3_2_conv_bias)

        with tf.name_scope("Pool") :
            max_pool_3 = tf.nn.max_pool(conv6,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    ### Left Branch 4th layer ###
    
    with tf.name_scope("Left_Branch_4th_Layer"):
        
        with tf.name_scope("Conv_1") :
            conv7 = tf.nn.conv2d(tf.pad(max_pool_3,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),left_4_1_conv,(1,3,3,1),padding = "VALID",name = "convolve")
            conv7 = tf.nn.bias_add(conv7,left_4_1_conv_bias,name = "bias_add")
            conv7 = tf.layers.batch_normalization(conv7,training = bool_train,name = "norm_7")
            conv7 =  tf.nn.leaky_relu(conv7,name = "activation")
            variable_summaries_weights_biases(left_4_1_conv)
            variable_summaries_weights_biases(left_4_1_conv_bias)
            
        with tf.name_scope("Conv_2") :
            conv8 = tf.nn.conv2d(tf.pad(conv7,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),left_4_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv8 = tf.nn.bias_add(conv8,left_4_2_conv_bias,name = "bias_add")
            conv8 = tf.layers.batch_normalization(conv8,training = bool_train,name = "norm_8")
            conv8 =  tf.nn.leaky_relu(conv8,name = "activation")
            variable_summaries_weights_biases(left_4_2_conv)
            variable_summaries_weights_biases(left_4_2_conv_bias)

        with tf.name_scope("Pool") :
            max_pool_4 = tf.nn.max_pool(conv8,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    
    ### Centre Branch ###
    
    with tf.name_scope("Centre_Branch"):
        
        with tf.name_scope("Conv_1") :
            
            conv9 = tf.nn.conv2d(tf.pad(max_pool_4,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),centre_5_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv9 = tf.nn.bias_add(conv9,centre_5_1_conv_bias,name = "bias_add")
            conv9 = tf.layers.batch_normalization(conv9,training = bool_train,name = "norm_9")
            conv9 =  tf.nn.leaky_relu(conv9,name = "activation")
            variable_summaries_weights_biases(centre_5_1_conv) 
            variable_summaries_weights_biases(centre_5_1_conv_bias)
            
        with tf.name_scope("Conv_2") :
        
            conv10 = tf.nn.conv2d(tf.pad(conv9,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),centre_5_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv10 = tf.nn.bias_add(conv10,centre_5_2_conv_bias,name = "bias_add")
            conv10 = tf.layers.batch_normalization(conv10,training = bool_train,name = "norm_10")
            conv10 =  tf.nn.leaky_relu(conv10,name = "activation")
            variable_summaries_weights_biases(centre_5_2_conv)
            variable_summaries_weights_biases(centre_5_2_conv_bias)

            conv10_obj = convolution(conv9.shape[1],conv9.shape[2],conv9.shape[3],centre_5_2_conv.shape[0],centre_5_2_conv.shape[1],centre_5_2_conv.shape[3],3,3,conv9.shape[1],conv9.shape[2])
            de_conv10_obj = trans_convolve(None,True,conv10_obj.output_h,conv10_obj.output_w,conv10_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d =128,stride_h = 2,stride_w = 2,padding = 'VALID')   
          
        with tf.name_scope("Deconvolve") : 
            de_conv10  = tf.nn.conv2d_transpose(conv10,centre_5_3_deconv, output_shape = (tf.shape(X)[0],de_conv10_obj.output_h,de_conv10_obj.output_w,de_conv10_obj.output_d), strides = (1,2,2,1),padding = 'VALID',name = "deconv")
            variable_summaries_weights_biases(centre_5_3_deconv)   

    ### Right Branch 4th layer ###
    
    with tf.name_scope("Merging") :
    
        merge1 = tf.concat([de_conv10,conv8],axis = 3,name = "merge")   
    
    
    with tf.name_scope("Right_Branch_4th_Layer"):
    
        with tf.name_scope("Conv_1") :
            
            conv11 = tf.nn.conv2d(tf.pad(merge1,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),right_4_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv11 = tf.nn.bias_add(conv11,right_4_1_conv_bias,name = "bias_add")
            conv11 = tf.layers.batch_normalization(conv11,training = bool_train,name = "norm_11")
            conv11 =  tf.nn.leaky_relu(conv11,name = "activation")
            variable_summaries_weights_biases(right_4_1_conv)

        with tf.name_scope("Conv_2") :
    
            conv12 = tf.nn.conv2d(tf.pad(conv11,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),right_4_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv12 = tf.nn.bias_add(conv12,right_4_2_conv_bias,name = "bias_add")
            conv12 = tf.layers.batch_normalization(conv12,training = bool_train,name = "norm_12")
            conv12 =  tf.nn.leaky_relu(conv12,name = "activation")
            variable_summaries_weights_biases(right_4_2_conv)
            variable_summaries_weights_biases(right_4_2_conv_bias)

            conv12_obj = convolution(conv11.shape[1],conv11.shape[2],conv11.shape[3],right_4_2_conv.shape[0],right_4_2_conv.shape[1],right_4_2_conv.shape[3],3,3,conv11.shape[1],conv11.shape[2])                
            de_conv12_obj = trans_convolve(None,True,conv12_obj.output_h,conv12_obj.output_w,conv12_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d = 256,stride_h = 2,stride_w = 2,padding = 'VALID')   
    
        with tf.name_scope("Deconvolve") :    
            de_conv12 = tf.nn.conv2d_transpose(conv12,right_4_3_deconv,output_shape = (tf.shape(X)[0],de_conv12_obj.output_h,de_conv12_obj.output_w,de_conv12_obj.output_d), strides = (1,2,2,1),padding = 'VALID',name = "deconv")
            variable_summaries_weights_biases(right_4_3_deconv)
    
    ### Right Branch 3rd layer ###
    
    with tf.name_scope("Merging") :
    
        merge2 = tf.concat([de_conv12,conv6],axis = 3,name = "merge")
    
    with tf.name_scope("Right_Branch_3rd_Layer"):
        
        with tf.name_scope("Conv_1") :
            
            conv13 = tf.nn.conv2d(tf.pad(merge2,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),right_3_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv13 = tf.nn.bias_add(conv13,right_3_1_conv_bias,name = "bias_add")
            conv13 = tf.layers.batch_normalization(conv13,training = bool_train,name = "norm_13")
            conv13 =  tf.nn.leaky_relu(conv13,name = "activation")
            variable_summaries_weights_biases(right_3_1_conv)    
            variable_summaries_weights_biases(right_3_1_conv_bias)
        
        with tf.name_scope("Conv_2") :
    
            conv14 = tf.nn.conv2d(tf.pad(conv13,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),right_3_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv14 = tf.nn.bias_add(conv14,right_3_2_conv_bias,name = "bias_add")
            conv14 = tf.layers.batch_normalization(conv14,training = bool_train,name = "norm_14")
            conv14 =  tf.nn.leaky_relu(conv14,name = "activation")        
            variable_summaries_weights_biases(right_3_2_conv)
            variable_summaries_weights_biases(right_3_2_conv_bias)
            conv14_obj = convolution(conv13.shape[1],conv13.shape[2],conv13.shape[3],right_3_2_conv.shape[0],right_3_2_conv.shape[1],right_3_2_conv.shape[3],3,3,conv13.shape[1],conv13.shape[2])                
            de_conv14_obj = trans_convolve(None,True,conv14_obj.output_h,conv14_obj.output_w,conv14_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d = 128,stride_h = 2,stride_w = 2,padding = 'VALID')
        
        with tf.name_scope("Deconvolve") :    
            de_conv14 = tf.nn.conv2d_transpose(conv14,right_3_3_deconv,output_shape = (tf.shape(X)[0],de_conv14_obj.output_h,de_conv14_obj.output_w,de_conv14_obj.output_d), strides = (1,2,2,1),padding = 'VALID',name = "deconv")
            variable_summaries_weights_biases(right_3_3_deconv)    
    
    ### Right Branch 2nd layer ###
    
    with tf.name_scope("Merging") :
        
        merge3 = tf.concat([de_conv14,conv4],axis = 3,name = "merge")
    
    with tf.name_scope("Right_Branch_2nd_Layer"):
  
        with tf.name_scope("Conv_1") :     
            conv15 = tf.nn.conv2d(tf.pad(merge3,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),right_2_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv15 = tf.nn.bias_add(conv15,right_2_1_conv_bias,name = "bias_add")
            conv15 = tf.layers.batch_normalization(conv15,training = bool_train,name = "norm_15")
            conv15 =  tf.nn.leaky_relu(conv15,name = "activation")
            variable_summaries_weights_biases(right_2_1_conv)
            variable_summaries_weights_biases(right_2_1_conv_bias)
            
        with tf.name_scope("Conv_2") :
            conv16 = tf.nn.conv2d(tf.pad(conv15,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),right_2_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv16 = tf.nn.bias_add(conv16,right_2_2_conv_bias,name = "bias_add")
            conv16 = tf.layers.batch_normalization(conv16,training = bool_train,name = "norm_16")
            conv16 = tf.nn.leaky_relu(conv16,name = "activation")
            variable_summaries_weights_biases(right_2_2_conv)
            variable_summaries_weights_biases(right_2_2_conv_bias)

            conv16_obj = convolution(conv15.shape[1],conv15.shape[2],conv15.shape[3],right_2_2_conv.shape[0],right_2_2_conv.shape[1],right_2_2_conv.shape[3],3,3,conv15.shape[1],conv15.shape[2])                    
            de_conv16_obj = trans_convolve(None,True,conv16_obj.output_h,conv16_obj.output_w,conv16_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d = 64,stride_h = 2,stride_w = 2,padding = 'VALID')    
           
        with tf.name_scope("Deconvolve") :
            de_conv16 = tf.nn.conv2d_transpose(conv16,right_2_3_deconv,output_shape = (tf.shape(X)[0],de_conv16_obj.output_h,de_conv16_obj.output_w,de_conv16_obj.output_d), strides = (1,2,2,1),padding = 'VALID',name = "deconv") 
            variable_summaries_weights_biases(right_2_3_deconv)
                    
    ### Right Branch 1st layer ###

    with tf.name_scope("Merging") :
        conv2 = tf.pad(conv2,paddings=[[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC')
        merge4 = tf.concat([de_conv16,conv2], axis = 3,name = "merge")


    with tf.name_scope("Right_Branch_1st_Layer"):

        with tf.name_scope("Conv1") : 
            conv17 = tf.nn.conv2d(merge4,right_1_1_conv,(1,1,1,1),padding = 'VALID',name = "convolve")
            conv17 = tf.nn.bias_add(conv17,right_1_1_conv_bias,name = "bias_add")  
            conv17 = tf.layers.batch_normalization(conv17,training = bool_train,name = "norm_17")
            conv17 = tf.nn.leaky_relu(conv17,name = "activation")
            variable_summaries_weights_biases(right_1_1_conv)
            variable_summaries_weights_biases(right_1_1_conv_bias)
            assert(conv17.shape[1:] == [120,120,32])
    
        with tf.name_scope("Conv2"):
            conv18 = tf.nn.conv2d(conv17,right_1_2_conv,(1,1,1,1),padding='VALID',name="convolve")
            conv18 = tf.nn.bias_add(conv18,right_1_2_conv_bias,name = "bias_add")
            conv18 = tf.layers.batch_normalization(conv18,training = bool_train,name = "norm_18")
            conv18 = tf.sigmoid(conv18,name="activation")
            variable_summaries_weights_biases(right_1_2_conv)
            variable_summaries_weights_biases(right_1_2_conv_bias)            
            assert(conv18.shape[1:] == [112,112,1])            
        
    return conv18

def compute_jaccard_cost(Y,Z3,batch_size) :
    ''' Computes the Jaccard Index and the Jaccard Cost.

    Description :
        The normal Jaccard index is non-differentiable. This function is an approximation to this index
        that makes this index differentiable.

    Arguments :
        Y          -- np.array.
                      Ground truth values of the mini-batch.
        Z3         -- np.array.
                      The ouput vector from the forward propagation.
        batch_size -- Int.
                      The number of images in a mini-batch.

    Returns :
        Jaccard      -- np.array.
                        Contains the jaccard values for each input image
        Jaccard_loss -- np.array.
                        Contains the jaccard loss for each input image.
    '''
    
    with tf.name_scope("Costs") :
        
        with tf.name_scope("Jaccard_Loss") :

            # Intersection
            nr = tf.multiply(Y,Z3)

            # Union
            dr = Y + Z3 -nr

            # Jaccard = Intersection/Union
            Jaccard = tf.divide( tf.reshape(tf.reduce_sum(nr,axis = [1,2,3]),shape = (batch_size,1)) , tf.reshape(tf.reduce_sum(dr,axis = [1,2,3] ),shape = (batch_size,1)) )
            
            Jaccard_loss = -tf.log(Jaccard)
            
            variable_summaries_weights_biases(Jaccard)
        
            return Jaccard,Jaccard_loss
        

############################################ NETWORK BUILDING ############################################

############################################# MODEL BUILDING #############################################

def model(epoch_num,img_rows,img_cols,num_channels,learning = 0.001,num_epochs = 100,batch_size = 16):

    # Tensorflow Graph
    X,Y = create_placeholders(img_rows,img_cols,num_channels)
    
    parameters = initialize_parameters()
    
    Z3 = forward_prop(X,parameters,bool_train = True)
    
    Jaccard,Jaccard_loss = compute_jaccard_cost(Y,Z3,batch_size)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = learning

    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10000,decay_rate = 1)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope("Optimizer") :
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Jaccard_loss,global_step=global_step,name = "Adam")
    # Tensorflow Graph
    
    init = tf.global_variables_initializer()
        
    # Creating the saving object 
    saver = tf.train.Saver(max_to_keep = 10000,var_list = tf.global_variables())
    
    merged = tf.summary.merge_all()
    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter("Summaries/Road_tar",sess.graph)
        
        sess.run(init)                
        
        path = os.path.join(os.getcwd(),"Parameters/Road_tar")
        ckpt = tf.train.get_checkpoint_state(path)
            
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoration of parameters of model with code_string Road_tar has been successfull")
            saver.restore(sess,ckpt.model_checkpoint_path)


        jaccard_list = []
        epoch_list = []
                
        for epoch in range(epoch_num,num_epochs) :
        
            print("Epoch Number : " + str(epoch))
            
            jaccards  = 0
            
            counting = 0
            for i in range(1500) :
                
                
                with open("./Data_training/Road/train" + "_" + str(i) + "_" + ".pkl","rb") as f :
                    X_input = pickle.load(f)
                
                with open("./Data_training/Road/test" + "_" + str(i) + "_" + ".pkl","rb") as f :
                    Y_input = pickle.load(f)

                if X_input is None or Y_input is None :
                    print("Something is wrong")
                    return None

                X_input = X_input/2047
                
                if ((epoch%1 == 0) and (counting == 1499)):
                    _,batch_jaccard,summary = sess.run([optimizer,Jaccard,merged], feed_dict = {X:X_input[:,:,:,0:9],Y:Y_input})                
                else:
                    _,batch_jaccard,learning_rate_val = sess.run([optimizer,Jaccard,learning_rate], feed_dict = {X:X_input[:,:,:,0:9],Y:Y_input})
                                
                X_input = None
                Y_input = None
                
                jaccards += np.sum(batch_jaccard)/batch_size
                
                counting += 1
                
            print(jaccards/1500)
            print(learning_rate_val)
            
            jaccard_list.append(jaccards/1500)
            epoch_list.append(epoch)
            
            if epoch%epoch_num == 0:
                highest_jaccard = jaccards 
            
            if epoch%1 == 0:
                
                train_writer.add_summary(summary,global_step = epoch)
                
                if highest_jaccard <= jaccards :
                    
                    highest_jaccard = jaccards
                
                    path = os.path.join(os.getcwd(),'Parameters/Road_tar/track-model.ckpt')
                    saver.save(sess,path,global_step=epoch)   
        
        train_writer.close()
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel("Jaccard")
        ax1.set_xlabel("Epochs")
        ax1.plot(epoch_list,jaccard_list)
        plt.show()
        plt.close()
        
        
############################################# MODEL BUILDING #############################################

if __name__ == '__main__':

    img_rows=112
    img_cols=112
    num_channels = 9
    epoch_num = 18

    model(epoch_num,img_rows,img_cols,num_channels)











