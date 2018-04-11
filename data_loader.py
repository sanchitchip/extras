import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd




class DataFolder(datasets):
    def __init__(self,root_folder,files,transforms = None):
    ## files:- csv file
        self.root_folder = root_folder
        self.image_path = list(files['image_dir'])


    def __len__(self):
        return(len(self.image_path))


    def __getitem__(self,index):
        image_path = self.image_path[index]
        image_a     = Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            image_a = self.transforms(image_a)
        return image_a




def get_loader(batch_size):
#    data_root = os.path.expanduser('~/.torch/data/mnist')
    
    files = pd.read_csv("/home/sanchit/PycharmProjects/sparse_data/flower102.csv")
    len1 = files.shape[0]

    nTrCls = np.random.choice(len1, 6000, replace = False)
    nTstCls = np.setdiff1d(np.arange(), nTrCls)
    train_samples = files.loc[files['index'].isin(nTrCls)]
    test_samples = files.loc[files['index'].isin(nTstCls)]
    vroot = "/home/sanchit/PycharmProjects/sparse_data/flower102/jpg"

    train_loader = DataLoader(DataFolder(root_folder = vroot,
                                         files = train_samples)
                              ,shuffle=True, batch_size=batch_size)
    
    test_loader = DataLoader(DataFolder(root_folder = vroot,
                                        files = train_samples)
                             , batch_size=batch_size)
    return train_loader, test_loader
