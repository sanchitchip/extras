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
##from VGGmodel import build_model
from matplotlib import pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

## Loading the data:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

## resize the image to inception model
## load the inception model
#base_model = build_model()

base_model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)



def preprocess_input(x):
#    pdb.set_trace()
    x = x/255
    x = x - 0.5
    x = x*2
    return x



inp = base_model.input
## all layer outputs
outputs = [base_model.get_layer("mixed"+ str(i)).output for i in range(11)]
#outputs = [base_model.get_layer("layer" + str(i)).output for i in range(1,14)]
#
functor = K.function([inp]+ [K.learning_phase()], outputs)


vlist = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

varray = 0
for i in range(2):
    pdb.set_trace()
    vimg_out = []
    vindex = np.where(y_train == i)
    new_arr = []

    for k in vindex[0]:
        vtemp = Image.fromarray(x_train[k])
        vtemp = vtemp.resize([299,299])
        vtemp = np.array(vtemp)    
        new_arr.append(vtemp)

    vobj_name = vlist[i]
    
    pdb.set_trace()

    for j in range(1000,6000,1000):
        layer_outs = functor([new_arr[j-1000:j], 1.])
        vmn = [np.mean(k, (0,3)) for k in layer_outs]
        vimg_out.append(vmn)

    pdb.set_trace()

    vimg_out = np.mean(vimg_out,0)
    if varray == 0:
        varray = [np.expand_dims(vimg_out[i],0)
                  for i in range(len(vimg_out))]
    else:
        varray = [np.concatenate((varray[i],
                                 np.expand_dims(vimg_out[i],0)),
                                 0)
                  for i in range(13)]

vsave_folder = "/home/sanchit/PycharmProjects/Cambridge/Inception/Result/" + "Subplot/"

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



###############################Extras################3


## to find the layer name
#for i in base_model.layers:
#    print(i.name)
#

## load cat image:

##pre-process
