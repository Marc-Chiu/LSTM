import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import tifffile as tiff
import os
from stardist.models import StarDist2D
from lamprogen.recipes.dl import stardist_2d_slicewise

modeldir = f'/Users/marcchiu/lamprogen-python/notebooks/deep learning/models'
name = 'lamprogen-stardist-trained'
model = StarDist2D(None, name=name,basedir=modeldir)



class DatasetLSTM(Dataset):
    def __init__(self, dataset_folder, im_size=[70, 1024, 1024], transform=None):

        self.__im = []
        self.__mask = []
        self.im_ht = im_size[0]
        self.im_wd = im_size[1]
        self.transform = transform

        # # Open and load text file including the whole training data
        im = dataset_folder + "/images"
        masks = dataset_folder + "/masks"
        
        for file in os.listdir(im):
            if file.lower().endswith('.tif'):
                img = tiff.imread(im + '/' + file)
                img_array = np.array(img)
                labeled = stardist_2d_slicewise(img_array, model)
                self.__im.append(labeled)
                break

        for file in os.listdir(masks):
            if file.lower().endswith('.tif'):
                img = tiff.imread(masks + '/' + file)
                img_array = np.array(img, dtype="float64")
                self.__mask.append(img_array)

    def __getitem__(self, idx):
        return self.__im[idx], self.__mask[idx]

    def __len__(self):

        return len(self.__im)