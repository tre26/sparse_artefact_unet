
import sys
import glob
import re
import os
import time
import numpy as np
import lasagne
from lasagne import layers
from lasagne.nonlinearities import rectify
from lasagne.init import GlorotUniform,Constant
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
floatX = theano.config.floatX
import random
import h5py 	
import matplotlib.pyplot as plt
import scipy.io as sio

def gen_cnn(input_var = None):
    inputLayer = layers.InputLayer(shape=(None, 1, 512, 512), input_var=input_var) 
    conv1 = layers.Conv2DLayer(inputLayer, num_filters= 64, filter_size=(3,3), stride=1, pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv1 = layers.Conv2DLayer(conv1, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    pool1 = layers.Pool2DLayer(conv1, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')
    conv2 = layers.Conv2DLayer(pool1, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv2 = layers.Conv2DLayer(conv2, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    pool2 = layers.Pool2DLayer(conv2, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')
    conv3 = layers.Conv2DLayer(pool2, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv3 = layers.Conv2DLayer(conv3, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    pool3 = layers.Pool2DLayer(conv3, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')
    conv4 = layers.Conv2DLayer(pool3, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv4 = layers.Conv2DLayer(conv4, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    pool4 = layers.Pool2DLayer(conv4, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')
    conv5 = layers.Conv2DLayer(pool4, num_filters= 1024, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv5 = layers.Conv2DLayer(conv5, num_filters= 1024, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    up6 = Unpool2DLayer(conv5, (2,2))
    conv6 = layers.Conv2DLayer(up6, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    merge6 = layers.ConcatLayer([conv6, conv4], axis = 1)
    conv6 = layers.Conv2DLayer(merge6, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv6 = layers.Conv2DLayer(conv6, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    up7 = Unpool2DLayer(conv6, (2,2))
    conv7 = layers.Conv2DLayer(up7, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    merge7 = layers.ConcatLayer([conv7, conv3], axis = 1)
    conv7 = layers.Conv2DLayer(merge7, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv7 = layers.Conv2DLayer(conv7, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    up8 = Unpool2DLayer(conv7, (2,2))
    conv8 = layers.Conv2DLayer(up8, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    merge8 = layers.ConcatLayer([conv8, conv2], axis = 1)
    conv8 = layers.Conv2DLayer(merge8, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    conv8 = layers.Conv2DLayer(conv8, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    up9 = Unpool2DLayer(conv8, (2,2))
    conv9 = layers.Conv2DLayer(up9, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    merge9 = layers.ConcatLayer([conv9, conv1], axis = 1)
    conv9 = layers.Conv2DLayer(merge9, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    vis25 = layers.get_output(conv9)
    conv9 = layers.Conv2DLayer(conv9, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(gain='relu'), b=Constant(0.), nonlinearity=rectify, flip_filters=True, convolution=conv2d)
    network = layers.Conv2DLayer(conv9, num_filters= 1, filter_size=(1,1), stride=(1, 1), pad='same', untie_biases=False, W=GlorotUniform(), b=Constant(0.), nonlinearity=None, flip_filters=True, convolution=conv2d)
    return network

class Unpool2DLayer(layers.Layer):

    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            ds = (ds, ds)
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must be an int or pair of int')
        self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)


tX = T.tensor4('inputs') 
tY = T.tensor3('targets') 

network = gen_cnn(tX)

tYhat = layers.get_output(network)
tYhat_test = layers.get_output(network, deterministic=True)

params = layers.get_all_params(network, trainable=True)

# demo code:

batchsize = 1

w_artifact_file = h5py.File('./test_32.mat','r') 
variables_w = w_artifact_file.items()

for var in variables_w:
    name_w_artifact = var[0] 
    data_w_artifact = var[1]
    if type(data_w_artifact) is h5py.Dataset:
        w_artifact = data_w_artifact.value 


wo_artifact_file = h5py.File('./test_GT.mat','r') 
variables_wo = wo_artifact_file.items()
for var in variables_wo:    
    name_wo_artifact = var[0] 
    data_wo_artifact = var[1]
    if type(data_wo_artifact) is h5py.Dataset:
        wo_artifact = data_wo_artifact.value 



if not os.path.exists('./result'):
    os.makedirs('./result')

saving_path = './result/'


loss = lasagne.objectives.squared_error(tYhat, tY).mean()
test_fn = theano.function([tX, tY], [tYhat_test, loss], on_unused_input='ignore') 

#load saved model

with np.load('./model/16843907') as n:
    param_values = [n['arr_%d' % j] for j in range(len(n.files))]
layers.set_all_param_values(network, param_values)


# test phase        
start_testing_t = time.time()

artifactual = []
GT = []
predicted = []

idc_test = range(wo_artifact.shape[0])


for b in range(0, len(idc_test), batchsize): 

    idcMB_test = idc_test[b : b+batchsize]

    X_test = np.expand_dims(np.require(w_artifact[idcMB_test, : ,:], dtype = floatX), axis = 0)

    Y_test = np.require(w_artifact[idcMB_test, :, :] - wo_artifact[idcMB_test, :, :], dtype = floatX)

    Yhattest, Ltest = test_fn(X_test , Y_test)

    artifactual.append(np.squeeze(np.squeeze(X_test, axis = 0)))
    GT.append(np.squeeze(np.squeeze(X_test, axis = 0) - Y_test))
    predicted.append(np.squeeze(np.squeeze(X_test, axis = 0) - np.squeeze(Yhattest, axis = 0)))

sio.savemat(saving_path + '/artifactual.mat', {"artifactual" : artifactual})
sio.savemat(saving_path + '/GT.mat', {"GT" : GT})
sio.savemat(saving_path + '/predicted.mat', {"predicted" : predicted})
        

print ("total testing took {:g} seconds".format(time.time() - start_testing_t))


