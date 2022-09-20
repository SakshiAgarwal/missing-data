import os
from numba import cuda
cuda.select_device(3)
print(cuda.current_context().get_memory_info())
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
from evaluate_helper import *

results=os.getcwd() + "/results/mnist-False-"
binary_data=False
ENCODER_PATH = "models/e_model_"+ str(binary_data) + ".pt"  ##without 20 is d=50
DECODER_PATH = "models/d_model_"+ str(binary_data) + ".pt"  ##simple is for simple VAE
ENCODER_PATH_UPDATED = "models/e_model_"+ str(binary_data) + "updated.pt" 
ENCODER_PATH_UPDATED_Test = "models/e_model_"+ str(binary_data) + "updated_test.pt" 

channels = 1    #1 for MNist
p = 28          # 28 for mnist
q = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = 50

#print(device)

encoder =  FlatWideResNet(channels=channels, size=1, levels=3, blocks_per_level=2, out_features = 2*d, shape=(p,q))
decoder = FlatWideResNetUpscaling(channels=channels, size=1, levels=3, blocks_per_level=2, in_features = d, shape=(p,q))
encoder_updated =  FlatWideResNet(channels=channels, size=1, levels=3, blocks_per_level=2, out_features = 2*d, shape=(p,q))
encoder_updated_test =  FlatWideResNet(channels=channels, size=1, levels=3, blocks_per_level=2, out_features = 2*d, shape=(p,q))

encoder = encoder.cuda()
decoder = decoder.cuda()
encoder_updated = encoder_updated.cuda()
encoder_updated_test = encoder_updated_test.cuda()

checkpoint = torch.load(ENCODER_PATH)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH)
decoder.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(ENCODER_PATH_UPDATED)
encoder_updated.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(ENCODER_PATH_UPDATED_Test)
encoder_updated_test.load_state_dict(checkpoint['model_state_dict'])

for params in encoder.parameters():
    params.requires_grad = False

for params in encoder_updated.parameters():
    params.requires_grad = False

for params in encoder_updated_test.parameters():
    params.requires_grad = False

for params in decoder.parameters():
    params.requires_grad = False

##Load parameters --
g_prior = True

if g_prior:
	file_save_params = results + str(-1) + "/pickled_files/params_mnist.pkl"
else:
	file_save_params = results + str(-1) + "/pickled_files/mixture_params_mnist.pkl"

with open(file_save_params, 'rb') as file:
	[pseudo_gibbs_sample,metropolis_gibbs_sample,z_params,iaf_params, mixture_params_inits,mixture_params,nb] = pickle.load(file)

with open(results + str(-1) + "/pickled_files/infereed_iwae_p_gaussian.pkl", 'rb') as file:
	[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
max_samples = 1000
x = np.arange(max_samples)
colours = ['g', 'b', 'y', 'r', 'k', 'c']

compare_iwae(lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples.png", ylim1= None, ylim2 = None)
exit()
test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, patches=True),batch_size=1)
#x = np.arange(max_samples)
#print(x)
z_iwae = np.zeros((max_samples))
iaf_iwae = np.zeros((max_samples))
mixture_iwae = np.zeros((max_samples))
mixture_inits_iwae = np.zeros((max_samples))
lower_bound = np.zeros((max_samples))
upper_bound = np.zeros((max_samples))
bound_updated_encoder = np.zeros((max_samples))
bound_updated_test_encoder = np.zeros((max_samples))
pseudo_gibbs_iwae = np.zeros((max_samples))
metropolis_within_gibbs_iwae = np.zeros((max_samples))
##Models trained with gaussian priors


if g_prior:
	p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
else:
	file_save = results + str(-1) + "/gmms.pkl"
	with open(file_save, 'rb') as file:
		gm = pickle.load(file)

	means_ = torch.from_numpy(gm.means_)
	std_ = torch.sqrt(torch.from_numpy(gm.covariances_))
	weights_ = torch.from_numpy(gm.weights_)
	p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(probs=weights_.cuda()), td.Independent(td.Normal(means_.cuda(), std_.cuda()), 1))


#evaluate 
print("Total", nb)
i=0
for data in test_loader:
	b_data, b_mask, b_full, labels = data

	lower_bound += eval_baseline(max_samples, p_z, encoder, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d)
	upper_bound += eval_baseline(max_samples, p_z, encoder, decoder, b_full.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d)
	bound_updated_encoder += eval_baseline(max_samples, p_z, encoder_updated, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d)
	bound_updated_test_encoder += eval_baseline(max_samples, p_z, encoder_updated_test, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d)

	pseudo_gibbs_iwae += evaluate_pseudo_gibbs(max_samples, p_z, encoder, decoder, pseudo_gibbs_sample[i].to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device)
	metropolis_within_gibbs_iwae += evaluate_pseudo_gibbs(max_samples, p_z, encoder, decoder, metropolis_gibbs_sample[i].to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device)

	z_iwae += evaluate_z(p_z = p_z, b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = z_params[i], decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples )

	iaf_iwae += evaluate_iaf(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),iaf_params = iaf_params[i], encoder = encoder, decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples )

	mixture_iwae +=  evaluate_z(p_z = p_z, b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = mixture_params[i], decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, ismixture=True )

	mixture_inits_iwae +=  evaluate_z(p_z = p_z, b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = mixture_params_inits[i], decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, ismixture=True )
	#print(z_iwae[-1],iaf_iwae[-1], mixture_iwae[-1],  mixture_inits_iwae[-1])

	i +=1
	if i%10==0:
		print(i)
		#break
		x = np.arange(max_samples)
		colours = ['g', 'b', 'y', 'r', 'k', 'c']

		if g_prior:
			compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples.png", ylim1= None, ylim2 = None)
		else:
			compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_mixture.png", ylim1= 0, ylim2 = 500)

		if g_prior:
			file_save_params = results + str(-1) + "/pickled_files/infereed_iwae_p_gaussian.pkl"
		else:
			file_save_params = results + str(-1) + "/pickled_files/infereed_iwae_p_mixture.pkl"

		with open(file_save_params, 'wb') as file:
			pickle.dump([lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i, nb], file)

	if i == 1000:
		break






