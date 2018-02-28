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

## os.environ["CUDA_VISIBLE_DEVICES"]="1"

def imgsave(img_path,index_val,varray,class_name):
#    pdb.set_trace()
    vfile_name = img_path + class_name + "layer" + str(index_val)
    plt.imsave(vfile_name,varray[i])
    return(print(vfile_name))


#unpickle-No USE NOW[As using form inbuilt cifar in keras]
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

img_save_folder ="/home/sanchit/PycharmProjects/Cambridge/Result/"
## no use as images taken directly from keras inbuilt lib.
img_path = "/home/sanchit/PycharmProjects/Cambridge/Image/cifar-10-batches-py/"


## creating the network and loading the weights.
base_model = build_model()
base_model.load_weights("/home/sanchit/PycharmProjects/Cambridge/cifar10vgg.h5")

## Loading the data:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


inp = base_model.input
# all layer outputs
outputs = [base_model.get_layer("layer" + str(i)).output for i in range(1,14)]

## to remove the flatten layers:-
#for i in [22,21,20,19]:
#    del outputs[i]
#
functor = K.function([inp]+ [K.learning_phase()], outputs)

# think of way to do it for all the class maybe using a for loop.

vlist = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

## to modify
vimg_out = []
for i in range(9):
    vindex = np.where(y_train == i)
    cat_data = x_train[vindex[0],:,:,:]
    vobj_name = vlist[i]
    for i in range(1000,6000,1000):
        layer_outs = functor([cat_data[i-1000:i,:,:,:], 1.])
        vmn = [np.mean(i, (0,3)) for i in layer_outs]
        vimg_out.append(vmn)
        for i in range(len(vmn)):
            imgsave(img_save_folder,i,vmn,vobj_name)

#pdb.set_trace()    
#vmn = [np.mean(i, (0,3)) for i in layer_outs]

