cuda_n = 1
import os
from numba import cuda
cuda.select_device(cuda_n)
print(cuda.current_context().get_memory_info())
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["NVIDIA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = str(cuda_n)
import torch
torch.cuda.set_device(cuda_n)
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

d = 1 #latent dim
batch_size = 64
learning_rate = 1e-3
num_epochs = 500 #2002
K=1
num_epochs_test = 300
dataset='banknote'

#50% random cells set to 0
train_miss, mask_train, valid_miss, mask_valid, test_miss, mask = load_uci_datasets(dataset=dataset, missing_for_train = True)
p = train_miss.shape[1]

results=os.getcwd() + "/results/uci_datasets/" + dataset + "/"
ENCODER_PATH = "models/" + dataset + "_e_model.pt" 
DECODER_PATH = "models/" + dataset + "_d_model.pt"  ##without 20 is d=50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

h = 128 # number of hidden units in (same for all MLPs)
d = 1 # dimension of the latent space
#K = 20 # number of IS during training
p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

decoder = nn.Sequential(
    torch.nn.Linear(d, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 2*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
)
encoder = nn.Sequential(
    torch.nn.Linear(p, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
)

encoder = encoder.cuda()
decoder = decoder.cuda()

checkpoint = torch.load(ENCODER_PATH, map_location='cuda:1')
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH, map_location='cuda:1')
decoder.load_state_dict(checkpoint['model_state_dict'])
print(torch.cuda.current_device())
print("model loaded")

for params in decoder.parameters():
    params.requires_grad = False

optimizer = torch.optim.Adam(list(encoder.parameters()), lr=learning_rate)

ENCODER_PATH_UPDATED = "models/" + dataset + "_e_model_updated.pt" 
ENCODER_PATH_UPDATED_TEST = "models/" + dataset + "_e_model_updated_test.pt"

## Only the encoder is updated
encoder1, decoder = train_VAE_uci(num_epochs, train_miss, valid_miss, ENCODER_PATH_UPDATED, results, encoder, decoder, optimizer, p_z, device, d, mask_train = mask_train, mask_valid=mask_valid)
encoder2, decoder = train_VAE_uci(num_epochs, test_miss, valid_miss, ENCODER_PATH_UPDATED_TEST, results, encoder, decoder, optimizer, p_z, device, d, mask_train = mask, mask_valid=mask_valid)



