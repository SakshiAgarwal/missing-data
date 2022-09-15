import os
from numba import cuda
cuda.select_device(3)
print(cuda.current_context().get_memory_info())
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["NVIDIA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import torch
torch.cuda.set_device(3)
print(torch.cuda.current_device())
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.distributions as td
from resnets import WideResNet, WideResNetUpscaling, FlatWideResNet, FlatWideResNetUpscaling
import os
from data import *
from loss import *
from train import *
from plot import *
from datetime import datetime
import gc
from inference import *
from networks import *
from init_methods import *
from pyro.nn import AutoRegressiveNN
from gmms import *
import pickle

"""
Initialize Hyperparameters
"""

d = 50 #latent dim
batch_size = 64
learning_rate = 1e-3
num_epochs = 2000
stop_early= False
binary_data = False
K=1
valid_size = 0.1
num_epochs_test = 300

#results="/home/sakshia1/myresearch/missing_dasta/miwae/pytorch/results/mnist-" + str(binary_data) + "-"
##results for beta-annealing
#results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-beta-annealing-"
##Results for alpha-annealing
results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-"
ENCODER_PATH = "models/e_model_"+ str(binary_data) + "ygivenx.pt"  ##without 20 is d=50

"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""

train_loader, val_loader = train_valid_loader(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data, return_labels=False)

#train_loader, val_loader = train_valid_loader_svhn(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data)
"""
Initialize the network and the Adam optimizer
"""

channels = 1  #1 for MNist
p = 28          # 28 for mnist
q = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

encoder =  FlatWideResNet(channels=channels, size=1, levels=3, blocks_per_level=2, out_features = 10, shape=(p,q))
#decoder = FlatWideResNetUpscaling(channels=1, size=1, levels=3, blocks_per_level=2, in_features = d, shape=(p,q))

encoder = encoder.cuda()
#decoder = decoder.cuda()
print(torch.cuda.current_device())

optimizer = torch.optim.Adam(list(encoder.parameters()), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""

#p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

best_loss, count = 0, 0

if num_epochs>0:
    encoder = train_pygivenx(num_epochs, train_loader, val_loader, ENCODER_PATH, results, encoder, optimizer, device, d, stop_early)
