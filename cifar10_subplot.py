#from keras.applications.resnet50 import ResNet50
from __future__ import print_function
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import vgg16
import keras.backend as K
import pdb
import os
from keras.datasets import cifar10
from VGGmodel import build_model
from matplotlib import pyplot as plt

## Loading the data:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

## creating the network and loading the weights.
base_model = build_model()
base_model.load_weights("/home/sanchit/PycharmProjects/Cambridge/cifar10vgg.h5")


inp = base_model.input
# all layer outputs
outputs = [base_model.get_layer("layer" + str(i)).output for i in range(1,14)]

functor = K.function([inp]+ [K.learning_phase()], outputs)


vlist = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

## aim is to make an array of layer_number,class_number,shape_of_layer

##pdb.set_trace()    
varray = 0
for i in range(10):
    vimg_out = []
    vindex = np.where(y_train == i)
    cat_data = x_train[vindex[0],:,:,:]
    vobj_name = vlist[i]
    for j in range(1000,6000,1000):
        layer_outs = functor([cat_data[j-1000:j,:,:,:], 1.])
        vmn = [np.mean(k, (0,3)) for k in layer_outs]
        vimg_out.append(vmn)
    vimg_out = np.mean(vimg_out,0)
    if varray == 0:
        varray = [np.expand_dims(vimg_out[i],0)
                  for i in range(len(vimg_out))]
    else:
        varray = [np.concatenate((varray[i],
                                 np.expand_dims(vimg_out[i],0)),
                                 0)
                  for i in range(13)]

vfile_name = "/home/sanchit/PycharmProjects/Cambridge/Result/"

vsave_folder = vfile_name + "Subplot2/"     

for i in range(13):
    # width, height in inches
    fig = plt.figure(figsize=(25, 25))
    for j in range(10):
##        pdb.set_trace()
        sub = fig.add_subplot(2, 5, j + 1)
        sub.set_title(vlist[j],size = 'xx-large')
        sub.imshow(varray[i][j], interpolation='nearest')
    plt.savefig(vsave_folder + "subplot_layer"+ str(i))
#

        
        
        
    




    
