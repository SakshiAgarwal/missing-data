import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from evaluate_helper import *

results=os.getcwd() + "/results/mnist-False-"
binary_data=False
##Load parameters --
g_prior = False
max_samples = 2000
patches = True

if patches:
	if g_prior:
		with open(results + str(-1) + "/pickled_files/infered_iwae_p_gaussian.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		file_name = "/compiled/IWAEvsSamples.png"
		ylim1 = None
		ylim2 = None
	else:
		with open(results + str(-1) + "/pickled_files/infereed_iwae_p_mixture.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		file_name = "/compiled/IWAEvsSamples_mixture.png"
		ylim1 = 0
		ylim2 = 500
else:
	if g_prior:
		with open(results + str(-1) + "/pickled_files/TH_mnist_iwae_p_gaussian.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		file_name = "/compiled/TH-IWAEvsSamples.png"
		ylim1 = None
		ylim2 = None
	else:
		with open(results + str(-1) + "/pickled_files/TH_mnist_iwae_p_mixture.pkl", 'rb') as file:
			[lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae, nb] = pickle.load(file)
		file_name = "/compiled/TH-IWAEvsSamples_mixture.png"
		ylim1= 500
		ylim2= 1000

x = np.arange(max_samples)
colours = ['g', 'b', 'y', 'r', 'k', 'c']

compare_iwae(lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, z_iwae, iaf_iwae, mixture_iwae , mixture_inits_iwae,  colours, x, "Estimated log-likelihood for missing pixels", results + str(-1) + file_name, ylim1= ylim1, ylim2 = ylim2)
