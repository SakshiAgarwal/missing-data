import os
from numba import cuda
import torch
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
from datetime import datetime
import gc
import pickle
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
import pyro
from mixture import *
from inference import *

def pass_decoder(K_samples, zgivenx, decoder, b_full, b_mask, channels, p, q,d, data):
	zgivenx_flat = zgivenx.reshape([K_samples,d])
	all_logits_obs_model = decoder.forward(zgivenx_flat)

	full_ = torch.Tensor.repeat(b_full,[K_samples,1,1,1]) 
	data_flat = full_.reshape([-1,1])

	mask_ = torch.Tensor.repeat(b_mask,[K_samples,1,1,1])  
	tiledmask = mask_.reshape([K_samples,channels*p*q]).cuda()

	if data=='mnist':
		all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
	else:
		all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale = sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
	#print(all_log_pxgivenz_flat)

	all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K_samples,channels*p*q])
	logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K_samples,1]) 
	logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K_samples,1]) 

	return logpxmissgivenz, all_logits_obs_model

def evaluate_z(p_z, b_data, b_full, b_mask, z_params, decoder, device, d, results, nb , K_samples, data='mnist', ismixture=False):
	if ismixture:
		[logits, means, scales] = z_params
		q_z = ReparameterizedNormalMixture1d(logits.to(device,dtype = torch.float), means.to(device,dtype = torch.float), torch.nn.Softplus()(scales.to(device,dtype = torch.float))) 
	else:
		q_z = td.Independent(td.Normal(loc=z_params[...,:d], scale=torch.nn.Softplus()(z_params[...,d:])),1)

	zgivenx = q_z.rsample([K_samples])
	channels = b_full.shape[1]
	p = b_full.shape[2]
	q = b_full.shape[3]

	logqz = q_z.log_prob(zgivenx).reshape(K_samples,1)
	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 

	logpxmissgivenz, all_logits_obs_model = pass_decoder(K_samples, zgivenx, decoder, b_full, b_mask, channels, p, q,d, data)

	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data

	#index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
	#predicted_image = b_data
	#predicted_image[~b_mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~b_mask])
	#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])
	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())

def eval_baseline(K_samples, p_z, encoder, decoder, iota_x, full, mask, d, data='mnist', with_labels=False):

	channels = full.shape[1]
	p = full.shape[2]
	q = full.shape[3]

	out_encoder = encoder.forward(iota_x)
	q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
	
	zgivenx = q_zgivenxobs.rsample([K_samples]).reshape(K_samples,d) 

	logqz = q_zgivenxobs.log_prob(zgivenx).reshape(K_samples,1)
	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 

	if with_labels:
		labels = torch.Tensor.repeat(labels,[K_samples,1]) 
		zgivenx_y = torch.cat((zgivenx,labels),1)
		zgivenx_flat = zgivenx_y.reshape([K_samples,d+10])
	else:
		zgivenx_flat = zgivenx.reshape([K_samples,d])

	logpxmissgivenz, all_logits_obs_model = pass_decoder(K_samples, zgivenx_flat, decoder, full, mask, channels, p, q,d, data)

	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data

	#index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
	#predicted_image = iota_x
	#predicted_image[~mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~mask])
	#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])
	#img = predicted_image.cpu().data.numpy()

	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())


def evaluate_pseudo_gibbs(K_samples, p_z, encoder, decoder, x_init, iota_x, full, mask, d, device, data='mnist', with_labels=False):
	channels = full.shape[1]
	p = full.shape[2]
	q = full.shape[3]

	iota_x[~mask] = x_init
	zgivenx = torch.zeros((K_samples,d)).to(device,dtype = torch.float)
	logpxmissgivenz = torch.zeros([K_samples,1]).to(device,dtype = torch.float)
	logqz = torch.zeros([K_samples,1]).to(device,dtype = torch.float)

	for i in range(K_samples):
		out_encoder = encoder.forward(iota_x)
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
		zgivenx[i] = q_zgivenxobs.rsample([1]).reshape(1,d) 
		logqz[i] = q_zgivenxobs.log_prob(zgivenx[i]).reshape(1,1)
		x_logits = decoder.forward(zgivenx[i])
		if data=='mnist':
			xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits.reshape(-1)),1)
		else:
			sigma_decoder = decoder.get_parameter("log_sigma")
			xgivenz = td.Normal(loc = x_logits.reshape([-1,1]), scale =  sigma_decoder.exp()*(torch.ones(*x_logits.shape).cuda()).reshape([-1,1]))

		iota_x[~mask] = xgivenz.sample().reshape(1,channels,p,q)[~mask]

	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 
	logpxmissgivenz, all_logits_obs_model = pass_decoder(K_samples, zgivenx, decoder, full, mask, channels, p, q, d, data)

	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data

	#index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
	#predicted_image = iota_x
	#predicted_image[~mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~mask])
	#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])

	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())


def evaluate_metropolis_within_gibbs(K_samples, p_z, encoder, decoder, x_init, iota_x, full, mask, d, device, data='mnist', with_labels=False):
	channels = full.shape[1]
	p = full.shape[2]
	q = full.shape[3]

	iota_x[~mask] = x_init
	zgivenx, logqz  = m_g_sampler(iota_x, full, mask, encoder, decoder, p_z, d, results=os.getcwd() + "/results/mnist-False-", nb=0, iterations=0, K=1, T=K_samples, data='mnist', evaluate=True)
	#logqz = q_zgivenxobs.log_prob(zgivenx).reshape(K_samples,1)
	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 
	logpxmissgivenz, all_logits_obs_model = pass_decoder(K_samples, zgivenx, decoder, full, mask, channels, p, q, d, data)

	logqz = logqz.reshape(K_samples,1)
	logpxmissgivenz = logpxmissgivenz.reshape(K_samples,1)
	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		#print(logpxmissgivenz[:i] + logpz[:i] - logqz[:i])
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data

	#index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
	#predicted_image = iota_x
	#predicted_image[~mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~mask])
	#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])

	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())


def evaluate_iaf(p_z, b_data, b_full, b_mask, iaf_params, encoder, decoder, device, d, results, nb , K_samples, data='mnist'):

	[t1, t2] = iaf_params
	autoregressive_nn =  AutoRegressiveNN(d, [320, 320]).cuda() .to(device,dtype = torch.float)
	autoregressive_nn2 =  AutoRegressiveNN(d, [320, 320]).cuda() .to(device,dtype = torch.float)
	autoregressive_nn.load_state_dict(t1)
	autoregressive_nn2.load_state_dict(t2)


	z_params = encoder.forward(b_data.to(device,dtype = torch.float))
	q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d], scale=torch.nn.Softplus()(z_params[...,d:])),1)

	transform = AffineAutoregressive(autoregressive_nn).cuda()
	transform2 = AffineAutoregressive(autoregressive_nn2).cuda()
	pyro.module("my_transform", transform)  
	flow_dist = pyro.distributions.torch.TransformedDistribution(q_zgivenxobs, [transform, transform2]) #, transform2, transform3, transform4, transform5
	
	zgivenx = flow_dist.rsample([K_samples])
	channels = b_full.shape[1]
	p = b_full.shape[2]
	q = b_full.shape[3]

	logqz = flow_dist.log_prob(zgivenx).reshape(K_samples,1)
	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 

	logpxmissgivenz, all_logits_obs_model  = pass_decoder(K_samples, zgivenx, decoder, b_full, b_mask, channels, p, q, d, data)

	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data


	#index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
	#predicted_image = b_data
	#predicted_image[~b_mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~b_mask])
	#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])
	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())


