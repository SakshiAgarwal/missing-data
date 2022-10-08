import os
from numba import cuda
cuda.select_device(1)
print(cuda.current_context().get_memory_info())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.cuda.set_device(1)
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

dataset='banknote'

trainset, validset, testset, mask, testset_full  = load_uci_datasets(dataset=dataset)

p = trainset.shape[1]

results=os.getcwd() + "/results/uci_datasets/" + dataset + "/"
ENCODER_PATH = "models/" + dataset + "_e_model.pt" 
DECODER_PATH = "models/" + dataset + "_d_model.pt"  
ENCODER_PATH_UPDATED = "models/" + dataset + "_e_model_updated.pt" 
ENCODER_PATH_UPDATED_TEST = "models/" + dataset + "_e_model_updated_test.pt"

h = 128 # number of hidden units in (same for all MLPs)
d = 1 # dimension of the latent space
#K = 20 # number of IS during training
p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_updated =  nn.Sequential(
    torch.nn.Linear(p, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
)

encoder_updated_test =  nn.Sequential(
    torch.nn.Linear(p, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
)

encoder = encoder.cuda()
decoder = decoder.cuda()
encoder_updated = encoder_updated.cuda()
encoder_updated_test = encoder_updated_test.cuda()

checkpoint = torch.load(ENCODER_PATH, map_location='cuda:1')
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH, map_location='cuda:1')
decoder.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(ENCODER_PATH_UPDATED, map_location='cuda:1')
encoder_updated.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(ENCODER_PATH_UPDATED_TEST, map_location='cuda:1')
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
write_only = True
to_plot = False
max_samples = 10000
g_prior = True

##Load saved files --
if g_prior:
	file_save_params = results + str(-1) + "/pickled_files/" + str(dataset) + "-params.pkl"
else:
	file_save_params = results + str(-1) + "/pickled_files/" + str(dataset) + "-params-mixture.pkl"

with open(file_save_params, 'rb') as file:
	[pseudo_gibbs_sample,metropolis_gibbs_sample,z_params,iaf_params, mixture_params_inits,mixture_params,nb] = pickle.load(file)


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
	file_save = os.getcwd()  + "/models/gmms_svhns.pkl"
	with open(file_save, 'rb') as file:
		gm = pickle.load(file)

	means_ = torch.from_numpy(gm.means_)
	std_ = torch.sqrt(torch.from_numpy(gm.covariances_))
	weights_ = torch.from_numpy(gm.weights_)
	p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(probs=weights_.cuda()), td.Independent(td.Normal(means_.cuda(), std_.cuda()), 1))


#evaluate 
print("Total", nb)
i=0

bs=1
n = np.shape(testset)[0]
perm = np.random.permutation(n)
batches_data = np.array_split(testset[perm,], n/bs)
batches_mask = np.array_split(mask[perm,], n/bs)
batches_full = np.array_split(testset_full[perm,], n/bs)

for it in range(len(batches_data)):
	#images = []
	i +=1
	if to_plot:
		if i< 20:
			continue

	b_data = torch.from_numpy(batches_data[it])
	b_mask = torch.from_numpy(batches_mask[it]).bool().cuda()
	b_full = torch.from_numpy(batches_full[it])
                
	#images.append(np.squeeze(b_full))
	#images.append(np.squeeze(b_data))

	lower = eval_baseline(max_samples, p_z, encoder, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='uci')
	lower_bound += lower
	#images.append(img)

	upper = eval_baseline(max_samples, p_z, encoder, decoder, b_full.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='uci')
	upper_bound += upper
	#images.append(img)

	updated_encoder = eval_baseline(max_samples, p_z, encoder_updated, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='uci')
	bound_updated_encoder += updated_encoder
	#images.append(img)

	updated_test_encoder  = eval_baseline(max_samples, p_z, encoder_updated_test, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='uci')
	bound_updated_test_encoder += updated_test_encoder
	#images.append(img)

	pseudo_gibbs_ = evaluate_pseudo_gibbs(max_samples, p_z, encoder, decoder, pseudo_gibbs_sample[i-1].to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device, data='uci')
	pseudo_gibbs_iwae += pseudo_gibbs_
	#images.append(img)

	metropolis_within_gibbs_ = evaluate_metropolis_within_gibbs(max_samples, p_z, encoder, decoder, metropolis_gibbs_sample[i-1].to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device, data='uci')
	metropolis_within_gibbs_iwae += metropolis_within_gibbs_
	#images.append(img)

	z_ = evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float),  b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = z_params[i-1].to(device,dtype = torch.float), decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, data='uci' )
	z_iwae += z_
	#images.append(img)

	iaf_= evaluate_iaf(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),iaf_params = iaf_params[i-1], encoder = encoder, decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, data='uci' )
	iaf_iwae += iaf_
	#images.append(img)

	mixture_ =  evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = mixture_params[i-1], decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, ismixture=True, data='uci' )
	mixture_iwae += mixture_
	#images.append(img)	

	mixture_inits_ =  evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = mixture_params_inits[i-1], decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, ismixture=True, data='uci' )
	mixture_inits_iwae += mixture_inits_
	#images.append(img)	

	if to_plot:
		if g_prior:
			plot_images_comparing_methods(images, file=results + str(-1) + "/compiled/"+ str(i)+"most_probable_imputations.png")
		else:
			plot_images_comparing_methods(images, file=results + str(-1) + "/compiled/"+ str(i)+"most_probable_imputations_mixture.png")
		if i==25:
			break	
			
	if write_only:
		if i%10==0:
			print(i)
			#break
			x = np.arange(max_samples)
			colours = ['g', 'b', 'y', 'r', 'k', 'c']

			if g_prior:   
				compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_"+ str(dataset) + ".png", ylim1= 0, ylim2 = 6)
				file_save_params = results + str(-1) + "/pickled_files/" + str(dataset) + "infered_iwae_p_gaussian.pkl"
			else:
				compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_"+ str(dataset) + "mixture.png", ylim1= None, ylim2 = None)
				file_save_params = results + str(-1) + "/pickled_files/" + str(dataset) + "infered_iwae_p_mixture.pkl"

			with open(file_save_params, 'wb') as file:
				pickle.dump([lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i, nb], file)
                
	print(lower_bound[-1]/i, upper_bound[-1]/i, bound_updated_encoder[-1]/i, bound_updated_test_encoder[-1]/i, pseudo_gibbs_iwae[-1]/i, metropolis_within_gibbs_iwae[-1]/i, z_iwae[-1]/i, iaf_iwae[-1]/i, mixture_iwae[-1]/i , mixture_inits_iwae[-1]/i)

	if i == 1001:
		break






