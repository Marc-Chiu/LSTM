import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from losses import DICELossMultiClass

from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import sys


#from data import DatasetLSTM
from CLSTM import CLSTM
from data import DatasetLSTM


from stardist.models import StarDist2D
from lamprogen.recipes.dl import stardist_2d_slicewise



## import stardist model in place of ku-net
modeldir = f'/Users/marcchiu/lamprogen-python/notebooks/deep learning/models'
name = 'lamprogen-stardist-trained'
model_stardist = StarDist2D(None, name=name,basedir=modeldir)

# load BDCLSTM Model
model = CLSTM(input_channels=1, hidden_channels=[64])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
criterion = DICELossMultiClass()
model = model.to(device)
criterion = criterion.to(device)


path = 'Fixed'
train_data = DatasetLSTM(path)
test_data = DatasetLSTM(path)

print(train_data)