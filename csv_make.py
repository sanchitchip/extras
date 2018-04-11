import glob
import pandas
import os
import numpy as np


cam1 = glob.glob("/home/sanchit/PycharmProjects/sparse_data/flower102/jpg/*.jpg")


columns = ['index','image_dir']
vindex = list(range(len(cam1)))

files = pandas.DataFrame(index=vindex, columns=columns)
files['index'] = vindex
files['image_dir'] = cam1

print(files.isnull().sum())
files.to_csv('/home/sanchit/PycharmProjects/sparse_data/flower102.csv', index=False)
