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

## image saver function
def imgsave(img_path,index_val,varray):
    vfile_name = img_path + "layer" + str(index_val)
    plt.imsave(vfile_name,varray)
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

# base VGG model:-
## images save folder:
img_save_folder ="/home/sanchit/PycharmProjects/Cambridge/Result/"
img_path = "/home/sanchit/PycharmProjects/Cambridge/Image/"

base_model = vgg16.VGG16(weights='imagenet')
layer_dict = dict([layers.name, layers] for layers in base_model.layers)
vimg = imgload(img_path)
vkeys = list(layer_dict.keys())
## image loading-> convert to function:-


for i in range(len(layer_dict)):
    if i> 18:
        break
    print(i)
    req_model = Model(inputs=base_model.input,
                      outputs=base_model.get_layer(vkeys[i]).output)
    vpred = req_model.predict(vimg)
    vmn = np.mean(vpred, (0,3))
    imgsave(img_save_folder,i,vmn)





from keras import backend as K

inp = base_model.input                                           # input placeholder
outputs = [layer.output for layer in base_model.layers]          # all layer outputs
## to remove the flatten layers:-
for i in [22,21,20,19]:
    del outputs[i]

functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

layer_outs = functor([vimg, 1.])

vmn = [np.mean(i, (0,3)) for i in layer_outs]

   

    
    
