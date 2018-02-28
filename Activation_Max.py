#from keras.applications.resnet50 import ResNet50
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
from VGGmodel import build_model,load_weight
## image saver function

def imgsave(img_path,index_val,varray):
#    pdb.set_trace()
    vfile_name = img_path + "layer" + str(index_val)
    plt.imsave(vfile_name,varray[i])
    return(print(vfile_name))

##img loaded function
def imgload(img_path):
    vlist = os.listdir(img_path)
    vimage = []
    for i in range(len(vlist)):
        vimg_name = img_path + vlist[i]
        img = image.load_img(vimg_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vimage.append(x)
        
    vimage = np.reshape(vimage, (len(vlist),224,224,3))    
    return vimage

#unpickle
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# base VGG model:-
## images save folder:
img_save_folder ="/home/sanchit/PycharmProjects/Cambridge/Result/"
img_path = "/home/sanchit/PycharmProjects/Cambridge/Image/cifar-10-batches-py/"

#base_model = vgg16.VGG16(weights='imagenet')
#loading VGG trained on cifar
base_model = build_model()
base_model.load_weights("/home/sanchit/PycharmProjects/Cambridge/cifar10vgg.h5")

vimg = imgload(img_path)

# input placeholder
inp = base_model.input
# all layer outputs
outputs = [layer.output for layer in base_model.layers]

## to remove the flatten layers:-
for i in [22,21,20,19]:
    del outputs[i]

functor = K.function([inp]+ [K.learning_phase()], outputs)
layer_outs = functor([vimg, 1.])
vmn = [np.mean(i, (0,3)) for i in layer_outs]
for i in range(len(vmn)):
    imgsave(img_save_folder,i,vmn)
    
