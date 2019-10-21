
import sys
import glob
import re
import os
import time
import numpy as np
import lasagne
from lasagne import layers
import theano
import theano.tensor as T
floatX = theano.config.floatX
import sklearn
import h5py 
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.measure import compare_ssim as ssim 

def unet(pretrained_weights = None,input_size = (batchsize, 1, 256, 256), learning_rate):

    tX = T.tensor4('inputs') 
    tY = T.tensor3('targets') 

    inputLayer = layers.InputLayer(shape=(batchsize, 1, 256, 256), input_var=tX) 

    conv1 = layers.Conv2DLayer(inputLayer, num_filters= 32, filter_size=(3,3), stride=1, pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    conv1 = layers.Conv2DLayer(conv1, num_filters= 32, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    vis1 = layers.get_output(conv1) # visualisation of the first layer?

    pool1 = layers.Pool2DLayer(conv1, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')

    conv2 = layers.Conv2DLayer(pool1, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    conv2 = layers.Conv2DLayer(conv2, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    pool2 = layers.Pool2DLayer(conv2, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')

    conv3 = layers.Conv2DLayer(pool2, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    conv3 = layers.Conv2DLayer(conv3, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    pool3 = layers.Pool2DLayer(conv3, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')

    conv4 = layers.Conv2DLayer(pool3, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    conv4 = layers.Conv2DLayer(conv4, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    up7 = Unpool2DLayer(conv4, (2,2))

    conv7 = layers.Conv2DLayer(up7, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    merge7 = layers.ConcatLayer([conv7, conv3], axis = 1)

    conv7 = layers.Conv2DLayer(merge7, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    conv7 = layers.Conv2DLayer(conv7, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    up8 = Unpool2DLayer(conv7, (2,2))

    conv8 = layers.Conv2DLayer(up8, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    merge8 = layers.ConcatLayer([conv8, conv2], axis = 1)

    conv8 = layers.Conv2DLayer(merge8, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    conv8 = layers.Conv2DLayer(conv8, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    up9 = Unpool2DLayer(conv8, (2,2))

    conv9 = layers.Conv2DLayer(up9, num_filters= 32, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    merge9 = layers.ConcatLayer([conv9, conv1], axis = 1)

    conv9 = layers.Conv2DLayer(merge9, num_filters= 32, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    conv9 = layers.Conv2DLayer(conv9, num_filters= 32, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    network = layers.Conv2DLayer(conv9, num_filters= 1, filter_size=(1,1), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=None, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

    #print "output shape:", network.output_shape
    loss = lasagne.objectives.squared_error(tYhat, tY).mean()
    tYhat = layers.get_output(network)
    tYhat_test = layers.get_output(network, deterministic=True)
    params = layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_model = theano.function([tX, tY], [tYhat, loss], updates=updates, on_unused_input='ignore')
         
    # TODO
    '''
    if(pretrained_weights):
        network.load_weights(pretrained_weights)
    '''
    
    print ('Number of network parameters: {:d}'.format(layers.count_params(network)))
    print ('Network architecture: '.format(lasagne2str(network)))
    
    return train_model
