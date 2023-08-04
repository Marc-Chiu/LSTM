import keras
import numpy as np
import tifffile as tiff

from stardist.models import StarDist2D
from lamprogen.recipes.dl import stardist_2d_slicewise

modeldir = f'models'
name = 'lamprogen-stardist-trained'
model = StarDist2D(None, name=name,basedir=modeldir)

class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    def __init__(self, image_IDs, mask_IDs, batch_size=5, dim=(70,1024,1024), n_channels=3, shuffle=True):
        #'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.mask_IDs = mask_IDs
        self.image_IDs = image_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_IDs) / self.batch_size))

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image_IDs_temp = [self.image_IDs[k] for k in indexes]
        mask_IDs_temp = [self.image_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_IDs_temp, mask_IDs_temp)

        return X, y

    # implement in a sec
    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_IDs_temp, mask_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))
        idx = [i for i in range(len(image_IDs_temp))]

        # Generate data
        for i, image_ID, mask_ID in zip(idx, image_IDs_temp, mask_IDs_temp):
            # Store sample
            image = tiff.imread('data/images/' + image_ID)
            image = np.expand_dims(image, axis=-1)
            X[i,] = image #stardist_2d_slicewise(image, model) / 255

            # Store mask
            mask = tiff.imread('data/images/' + mask_ID) / 255
            mask = np.expand_dims(mask, axis=-1)
            y[i,] = mask

        return X, y