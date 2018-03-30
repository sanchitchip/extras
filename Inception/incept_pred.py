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

##
vcat = "/home/sanchit/Downloads/animal-animal-photography-cat-104827.jpg"
vcat = "/home/sanchit/Downloads/african-elephant-bull.jpg"
vim = Image.open(vcat)
var1 = vim.resize([299,299])
var2 = np.array(var1)
var3 = preprocess_input(var2)
vpred = base_model.predict(np.expand_dims(var3,0))
vout = decode_predictions(vpred)[0][0][1]


vlist = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

##
vindex = np.where(y_train == 0)
new_arr = []
for i in range(len(vindex[0])):
    if i%100 == 0:
        print(i)
        vtemp = Image.fromarray(x_train[vindex[0][i]])
        vtemp = vtemp.resize([299,299])
        vtemp = np.array(vtemp)
        vtemp = preprocess_input(vtemp)
        new_arr.append(vtemp)


vpred = []
for i in range(len(new_arr)):
    print(i)
    vtemp = base_model.predict(np.expand_dims(new_arr[i],0))
    vtemp = decode_predictions(vtemp)[0][0][1]
    vpred.append(vtemp)
    
    
    
