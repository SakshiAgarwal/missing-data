import os
from numba import cuda
cuda.select_device(0)
print(cuda.current_context().get_memory_info())
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import torch
torch.cuda.set_device(0)
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

results=os.getcwd() + "/results/svhn/" 
##Load parameters --
top_half = True
g_prior = True
write_only = True
to_plot = False
max_samples = 1000

if top_half:
    ENCODER_PATH_UPDATED = "models/svhn_e_model_TH-updated.pt" 
    ENCODER_PATH_UPDATED_Test = "models/svhn_e_model_TH-updated_test.pt" 
else:
    ENCODER_PATH_UPDATED = "models/svhn_e_model_patches-updated.pt" 
    ENCODER_PATH_UPDATED_Test = "models/svhn_e_model_patches-updated_test.pt" 

ENCODER_PATH = "models/svhn_encoder_anneal.pth" 
DECODER_PATH = "models/svhn_decoder_anneal.pth" 

channels = 3    #1 for MNist
p = 32          # 28 for mnist
q = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = 50

#print(device)

encoder =  FlatWideResNet(channels=channels, size=2, levels=3, dense_blocks=2, out_features = 2*d, activation=nn.LeakyReLU(), shape=(p,q)) #blocks_per_level=2,
decoder = FlatWideResNetUpscaling(channels=channels, size=2, levels=3, dense_blocks=2, transpose=True, activation=nn.LeakyReLU(), in_features = d, shape=(p,q), model ='sigma_vae', evaluate=True)

   
encoder_updated =  FlatWideResNet(channels=channels, size=2, levels=3, dense_blocks=2, out_features = 2*d, activation=nn.LeakyReLU(), shape=(p,q))
encoder_updated_test = FlatWideResNet(channels=channels, size=2, levels=3, dense_blocks=2, out_features = 2*d, activation=nn.LeakyReLU(), shape=(p,q))

encoder = encoder.cuda()
decoder = decoder.cuda()
encoder_updated = encoder_updated.cuda()
encoder_updated_test = encoder_updated_test.cuda()

checkpoint = torch.load(ENCODER_PATH, map_location='cuda:0')
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH, map_location='cuda:0')
decoder.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(ENCODER_PATH_UPDATED, map_location='cuda:0')
encoder_updated.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(ENCODER_PATH_UPDATED_Test, map_location='cuda:0')
encoder_updated_test.load_state_dict(checkpoint['model_state_dict'])

for params in encoder.parameters():
    params.requires_grad = False

for params in encoder_updated.parameters():
    params.requires_grad = False

for params in encoder_updated_test.parameters():
    params.requires_grad = False

for params in decoder.parameters():
    params.requires_grad = False
    
    
encoder.eval()
decoder.eval()
encoder_updated.eval()
encoder_updated_test.eval()


##Load saved files --
if top_half:
	if g_prior:
		file_save_params = results + str(-1) + "/pickled_files/params_svhn_TH.pkl"
	else:
		file_save_params = results + str(-1) + "/pickled_files/params_svhn_TH_mixture.pkl"
	test_loader = torch.utils.data.DataLoader(dataset=SVHN_Test(top_half=True),batch_size=1)
else:
	if g_prior:
		file_save_params = results + str(-1) + "/pickled_files/params_svhn_patches.pkl"
	else:
		file_save_params = results + str(-1) + "/pickled_files/params_svhn_patches_mixture.pkl"
	test_loader = torch.utils.data.DataLoader(dataset=SVHN_Test(patches=True),batch_size=1)

with open(file_save_params, 'rb') as file:
	[pseudo_gibbs_sample,metropolis_gibbs_sample,z_params,iaf_params, mixture_params_inits,mixture_params,nb] = pickle.load(file)

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

count = 0
for data in test_loader:
	images = []
	i +=1
	if to_plot:
		if i< 70:
			continue
		if i>80:
			exit()
	print("Image : ", i%10)            
	b_data, b_mask, b_full = data
	plot_image_svhn(np.squeeze(b_full.cpu().data.numpy()), results + str(-1) + "/compiled/" + str(i%10) +  "true.png")
	#continue

	#images.append(np.squeeze(b_full))
	#images.append(np.squeeze(b_data))

	#lower = eval_baseline(max_samples, p_z, encoder, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn')
	#lower_bound += lower
	#images.append(img)

	upper = eval_baseline(max_samples, p_z, encoder, decoder, b_full.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn')
	upper_bound += upper
	print("Upper bound: ", upper[-1])    
	if upper[-1]>600:
		continue        
	#images.append(img)
	count +=1

	updated_encoder = eval_baseline(max_samples, p_z, encoder_updated, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn',  nb=i)
	bound_updated_encoder += updated_encoder
	images.append(img)
	print("updated encoder (train): ", updated_encoder[-1])


	updated_test_encoder, img  = eval_baseline(max_samples, p_z, encoder_updated_test, decoder, b_data.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn', to_plot=True, nb=i, test=True)
	bound_updated_test_encoder += updated_test_encoder
	images.append(img)
	print("updated encoder (test): ", updated_test_encoder[-1])


	#pseudo_gibbs_ = evaluate_pseudo_gibbs(max_samples, p_z, encoder, decoder, pseudo_gibbs_sample[i-1].to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device, data='svhn')
	#pseudo_gibbs_iwae += pseudo_gibbs_
	#images.append(img)

	#metropolis_within_gibbs_ = evaluate_metropolis_within_gibbs(max_samples, p_z, encoder, decoder, metropolis_gibbs_sample[i-1].to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device, data='svhn')
	#metropolis_within_gibbs_iwae += metropolis_within_gibbs_
	#images.append(img)

	#z_ = evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float),  b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = z_params[i-1].to(device,dtype = torch.float), decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, data='svhn' )
	#z_iwae += z_
	#images.append(img)

	#iaf_= evaluate_iaf(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),iaf_params = iaf_params[i-1], encoder = encoder, decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, data='svhn' )
	#iaf_iwae += iaf_
	#images.append(img)

	#mixture_ =  evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = mixture_params[i-1], decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, ismixture=True, data='svhn' )
	#mixture_iwae += mixture_
	#images.append(img)	

	mixture_inits_, img =  evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = mixture_params_inits[i-1], decoder = decoder, device = device, d = d, results = results, nb=i , K_samples = max_samples, ismixture=True, data='svhn', to_plot=True, do_random=True )
	mixture_inits_iwae += mixture_inits_
	images.append(img)
	print("mixture (re-inits): ", mixture_inits_[-1])
    
	if to_plot:
		if g_prior:
			plot_images_comparing_methods(images, file=results + str(-1) + "/compiled/"+ str(i)+"most_probable_imputations.png", data='svhn')
		else:
			plot_images_comparing_methods(images, file=results + str(-1) + "/compiled/"+ str(i)+"most_probable_imputations_mixture.png", data='svhn')
		#if i==25:
		#	break
			
	if write_only:
		if i%10==0:
			print(i)
			#break
			x = np.arange(max_samples)
			colours = ['g', 'b', 'y', 'r', 'k', 'c']

			if top_half:
				if g_prior:   
					compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_svhn_TH_low-ll.png", ylim1= None, ylim2 = None)
					file_save_params = results + str(-1) + "/pickled_files/svhn_TH_infered_iwae_p_gaussian_low-ll.pkl"
				else:
					compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_svhn_TH_mixture_low-ll.png", ylim1= None, ylim2 = None)
					file_save_params = results + str(-1) + "/pickled_files/svhn_TH_infereed_iwae_p_mixture_low-ll.pkl"
			else:
				if g_prior:
					compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_svhn_patches_low-ll.png", ylim1= None, ylim2 = None)
					file_save_params = results + str(-1) + "/pickled_files/svhn_patches_infered_iwae_p_gaussian_low-ll.pkl"
				else:
					compare_iwae(lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_svhn_patches_mixture_low-ll.png", ylim1= None, ylim2 = None)
					file_save_params = results + str(-1) + "/pickled_files/svhn_patches_infereed_iwae_p_mixture_low-ll.pkl"

			with open(file_save_params, 'wb') as file:
				pickle.dump([lower_bound/i, upper_bound/i, bound_updated_encoder/i, bound_updated_test_encoder/i, pseudo_gibbs_iwae/i, metropolis_within_gibbs_iwae/i, z_iwae/i, iaf_iwae/i, mixture_iwae/i , mixture_inits_iwae/i, nb], file)


	if i == 1001:
		break
        
print(count)






