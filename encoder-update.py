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

d = 50 #latent dim
num_epochs = 100
stop_early= False
binary_data = False
K=1
valid_size = 0.1
num_epochs_test = 300

##Change here -- 
data = 'svhn'
top_half = True

if data =='mnist':
	batch_size = 64
	learning_rate = 1e-3

	results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-"
	ENCODER_PATH = "models/e_model_"+ str(binary_data) + ".pt"  ##without 20 is d=50
	DECODER_PATH = "models/d_model_"+ str(binary_data) + ".pt"  ##simple is for simple VAE
	#top-half
	train_loader, val_loader = train_valid_loader(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data, top_half=True)

else:
	batch_size = 128
	learning_rate = 3e-4
	results=os.getcwd() + "/results/svhn/" 
	ENCODER_PATH = "models/svhn_encoder_anneal.pth"  
	DECODER_PATH = "models/svhn_decoder_anneal.pth"
	if top_half:
		train_loader, val_loader = train_valid_loader_svhn(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data, top_half=True)
	else:
		train_loader, val_loader = train_valid_loader_svhn(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data, patches=True)

##Patches of missing data in train & valid data
#train_loader, val_loader = train_valid_loader(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data, ispatches=True)

##Top-half missing


#test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, patches=True),batch_size=64)

if data =='mnist':
	test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, top_half=True),batch_size=64)
	channels = 1    #1 for MNist
	p = 28          # 28 for mnist
	q = 28
else:
	if top_half:
		test_loader = torch.utils.data.DataLoader(dataset=SVHN_Test(top_half=True),batch_size=)
	else:
		test_loader = torch.utils.data.DataLoader(dataset=SVHN_Test(patches=True),batch_size=1)
	channels = 3    #1 for MNist
	p = 32        # 28 for mnist
	q = 32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if data =='mnist':
	encoder =  FlatWideResNet(channels=channels, size=1, levels=3, blocks_per_level=2, out_features = 2*d, shape=(p,q))
	decoder = FlatWideResNetUpscaling(channels=channels, size=1, levels=3, blocks_per_level=2, in_features = d, shape=(p,q))
else:
	encoder =  FlatWideResNet(channels=channels, size=2, levels=3, dense_blocks=2, out_features = 2*d, activation=nn.LeakyReLU(), shape=(p,q)) #blocks_per_level=2,
	decoder = FlatWideResNetUpscaling(channels=channels, size=2, levels=3, dense_blocks=2, transpose=True, activation=nn.LeakyReLU(), in_features = d, shape=(p,q), model ='sigma_vae')

encoder = encoder.cuda()
decoder = decoder.cuda()

checkpoint = torch.load(ENCODER_PATH)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH)
decoder.load_state_dict(checkpoint['model_state_dict'])
print(torch.cuda.current_device())
print("model loaded")

optimizer = torch.optim.Adam(list(encoder.parameters()), lr=learning_rate)

if data == 'svhn':
	if top_half:
		ENCODER_PATH1 = "models/svhn_e_model_TH-updated.pt" 
		ENCODER_PATH2 = "models/svhn_e_model_TH-updated_test.pt" 
	else:
		ENCODER_PATH1 = "models/svhn_e_model_patches-updated.pt" 
		ENCODER_PATH2 = "models/svhn_e_model_patches-updated_test.pt" 
else:
	ENCODER_PATH = "models/e_model_"+ str(binary_data) + "top_half-updated-test.pt" 

decoder.eval()

p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
## Only the encoder is updated

if data=='mnist':
	encoder, decoder = train_VAE(num_epochs, test_loader, val_loader, ENCODER_PATH, results, encoder, decoder, optimizer, p_z, device, d, stop_early)
else:
    encoder1, decoder1 = train_VAE_SVHN(num_epochs, train_loader, val_loader, ENCODER_PATH1, results, encoder, decoder, optimizer, p_z, device, d, stop_early, annealing=True)
    encoder2, decoder2 = train_VAE_SVHN(num_epochs, test_loader, val_loader, ENCODER_PATH2, results, encoder, decoder, optimizer, p_z, device, d, stop_early, annealing=True)




