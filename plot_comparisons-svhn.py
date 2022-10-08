import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from evaluate_helper import *

results=os.getcwd() + "/results/svhn/"
binary_data=False
##Load parameters --
g_prior = True
max_samples = 10000
patches = False

if patches:
	if g_prior:
		with open(results + str(-1) + "/pickled_files/svhn_patches_infered_iwae_p_gaussian.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		with open(results + str(-1) + "/pickled_files/svhn_patches_infered_iwae_p_gaussian_UE.pkl", 'rb') as file:
			[lower_bound_eu, upper_bound_eu, bound_updated_encoder_eu, bound_updated_test_encoder_eu, pseudo_gibbs_iwae_eu, metropolis_within_gibbs_iwae_eu, z_iwae_eu, iaf_iwae_eu, mixture_iwae_eu , mixture_inits_iwae_eu, nb] = pickle.load(file)
		file_name = "/compiled/IWAEvsSamples_svhn_patches.png"
		ylim1 = -2000
		ylim2 = -500
		#ylim1 = None
		#ylim2 = None
	else:
		with open(results + str(-1) + "/pickled_files/svhn_patches_infereed_iwae_p_mixture.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		file_name = "/compiled/IWAEvsSamples_svhn_patches_mixture.png"
		with open(results + str(-1) + "/pickled_files/svhn_patches_infereed_iwae_p_mixture_UE.pkl", 'rb') as file:
			[lower_bound_eu, upper_bound_eu, bound_updated_encoder_eu, bound_updated_test_encoder_eu, pseudo_gibbs_iwae_eu, metropolis_within_gibbs_iwae_eu, z_iwae_eu, iaf_iwae_eu, mixture_iwae_eu , mixture_inits_iwae_eu, nb] = pickle.load(file)
		ylim1 = -2000
		ylim2 = -500
		#ylim1 = None
		#ylim2 = None
else:
	if g_prior:
		with open(results + str(-1) + "/pickled_files/svhn_TH_infered_iwae_p_gaussian.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		#for i in range(len(upper_bound)):  
		#print(np.argwhere(upper_bound[] > 0.01))            
		file_name = "/compiled/IWAEvsSamples_svhn_TH.png"
		ylim1 = -6500
		ylim2 = -3000
		#ylim1 = None
		#ylim2 = None
	else:
		with open(results + str(-1) + "/pickled_files/svhn_TH_infereed_iwae_p_mixture.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		with open(results + str(-1) + "/pickled_files/svhn_TH_infereed_iwae_p_mixture_UE.pkl", 'rb') as file:
			[lower_bound_eu, upper_bound_eu, bound_updated_encoder_eu, bound_updated_test_encoder_eu, pseudo_gibbs_iwae_eu, metropolis_within_gibbs_iwae_eu, z_iwae_eu, iaf_iwae_eu, mixture_iwae_eu , mixture_inits_iwae_eu, nb] = pickle.load(file)
  
		file_name = "/compiled/IWAEvsSamples_svhn_TH_mixture.png"
		ylim1 = -11000
		ylim2 = -3000
		#ylim1 = None
		#ylim2 = None

x = np.arange(max_samples)
colours = ['g', 'b', 'y', 'r', 'k', 'c']

compare_iwae(lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae,  colours, x, "Estimated log-likelihood for missing pixels", results + str(-1) + file_name, ylim1= ylim1, ylim2 = ylim2)
