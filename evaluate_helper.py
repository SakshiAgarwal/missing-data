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
from inference_uci_datasets import *


def m_g_sampler(iota_x, full, mask, encoder, decoder, p_z, d, results, nb, iterations, K=1, T=1000, data='mnist', evaluate=False, with_labels=False, labels= None):
	batch_size = iota_x.shape[0]
	channels = iota_x.shape[1]
	p = iota_x.shape[2]
	q = iota_x.shape[3]
	i = -1

	if with_labels:
		x_prev = iota_x[0,0,:,:].reshape([1,1,28,28])[~mask]
	else:
		x_prev = iota_x[~mask]
	#z_prev = p_z.rsample([1])
	out_encoder = encoder.forward(iota_x)

	if data=='mnist':
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
	else:
		scales_z = 1e-2 + torch.nn.Softplus()(out_encoder[...,d:])
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=scales_z),1)

	z_prev = q_zgivenxobs.rsample([K])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 

	m_loglikelihood = []
	m_nelbo = []
	m_error = []
	#print(T)

	do_plot = False
	interval = int(T/4)
	#print(torch.cuda.current_device())

	zgivenx_evaluate = torch.zeros((T,d)).to(device,dtype = torch.float)
	logqz_evaluate = torch.zeros((T,1)).to(device,dtype = torch.float)

	for t in range(T):
		#print("inside metropolis within gibbs:", torch.mean(iota_x[:,0,:,:][~mask[:,0,:,:]]), torch.mean(iota_x[:,1,:,:][~mask[:,1,:,:]]), torch.mean(iota_x[:,2,:,:][~mask[:,2,:,:]]))
		if with_labels:
			iota_x[0,0,:,:].reshape([1,1,28,28])[~mask] = x_prev
		else:
			iota_x[~mask] = x_prev

		out_encoder = encoder.forward(iota_x)

		if data=='mnist':
			q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
		else:
			scales_z = 1e-2 + torch.nn.Softplus()(out_encoder[...,d:])
			q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=scales_z),1)


		z_t = q_zgivenxobs.rsample([K])
		zgivenx_evaluate[t] = z_t

		p_z_t = p_z.log_prob(z_t)
		p_z_prev = p_z.log_prob(z_prev)

		q_z_t = q_zgivenxobs.log_prob(z_t)
		logqz_evaluate[t] = q_z_t
		q_z_prev = q_zgivenxobs.log_prob(z_prev)

		if with_labels:
			labels = torch.zeros(1,1,10).to(device,dtype = torch.float)
			zgivenx_y = torch.cat((z_t,labels),2)
			zgivenx_flat = zgivenx_y.reshape([1,d+10])

			zgivenx_y_prev = torch.cat((z_prev,labels),2)
			zgivenx_flat_prev = zgivenx_y_prev.reshape([1,d+10])
		else:
			zgivenx_flat = z_t.reshape([K*batch_size,d])
			zgivenx_flat_prev = z_prev.reshape([K*batch_size,d])

		all_logits_obs_model = decoder.forward(zgivenx_flat)
		all_logits_obs_model_prev = decoder.forward(zgivenx_flat_prev)


		if with_labels:
			iota_x_flat = iota_x[0,0,:,:].reshape(batch_size,1*p*q)
			data_flat = iota_x[0,0,:,:].reshape([-1,1]).cuda()
			tiledmask = mask.reshape([batch_size,1*p*q]).cuda()
		else:
			iota_x_flat = iota_x.reshape(batch_size,channels*p*q)
			data_flat = iota_x.reshape([-1,1]).cuda()
			tiledmask = mask.reshape([batch_size,channels*p*q]).cuda()

		if data=='mnist':
			all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
			all_log_pxgivenz_flat_prev = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model_prev.reshape([-1,1])).log_prob(data_flat)
		else:
			sigma_decoder = decoder.get_parameter("log_sigma")
			scales_z = 1e-3 + torch.nn.Softplus()(sigma_decoder)
			all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale =  scales_z*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
			all_log_pxgivenz_flat_prev = td.Normal(loc = all_logits_obs_model_prev.reshape([-1,1]), scale =  scales_z*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)

		if with_labels:
			channels=1

		all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])
		all_log_pxgivenz_prev = all_log_pxgivenz_flat_prev.reshape([K*batch_size,channels*p*q])

		logpxobsgivenz = torch.sum(all_log_pxgivenz,1).reshape([K,batch_size]) #*tiledmask
		logpxobsgivenz_all = torch.sum(all_log_pxgivenz,1).reshape([K,batch_size])

		logpxobsgivenz_prev = torch.sum(all_log_pxgivenz_prev,1).reshape([K,batch_size]) #*tiledmask
		logpxobsgivenz_prev_all = torch.sum(all_log_pxgivenz_prev,1).reshape([K,batch_size])

		logpz = p_z.log_prob(z_t)
		logpz_prev = p_z.log_prob(z_prev)

		logq = q_zgivenxobs.log_prob(z_t)
		logq_prev = q_zgivenxobs.log_prob(z_prev)

		if data=='mnist':
			xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model),1)
		else:
			scales_z = 1e-3 + torch.nn.Softplus()(sigma_decoder)
			xgivenz = td.Normal(loc = all_logits_obs_model, scale =  scales_z*(torch.ones(*all_logits_obs_model.shape).cuda()))

		#xgivenz = td.Independent(td.bernoulli.Bernoulli(logits=all_logits_obs_model),1)

		p_data_t = logpxobsgivenz
		p_data_prev = logpxobsgivenz_prev

		log_rho = p_data_t + p_z_t + q_z_prev - p_data_prev - p_z_prev - q_z_t

		if log_rho>torch.tensor(0):
			log_rho=torch.tensor(0).to(device,dtype = torch.float)

		a = torch.rand(1).to(device,dtype = torch.float)
		#v = torch.tensor()
		if a < torch.exp(log_rho):
			z_prev = z_t
			x_prev = xgivenz.sample().reshape(batch_size,channels,p,q)[~mask]

			v = all_logits_obs_model.reshape([batch_size,channels,p,q])[~mask]
			neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))

			loglike = torch.mean(torch.sum(logpz + logpxobsgivenz_all,0))
			m_nelbo.append(neg_bound.item())
			#m_loglikelihood.append(loglike.item())
			#print("accept ----")
			xm_logits = all_logits_obs_model.reshape([batch_size,channels,p,q])
		else:
			if data=='mnist':
				xgivenz_prev = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model_prev),1)
			else:
				scales_z = 1e-3 + torch.nn.Softplus()(sigma_decoder)
				xgivenz_prev = td.Normal(loc = all_logits_obs_model_prev, scale =  scales_z*(torch.ones(*all_logits_obs_model_prev.shape).cuda()))

			x_prev = xgivenz_prev.sample().reshape(batch_size,channels,p,q)[~mask]

			v = all_logits_obs_model_prev.reshape([batch_size,channels,p,q])[~mask]
			neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz_prev + logpz_prev - logq_prev,0))
			loglike = torch.mean(torch.sum(logpz_prev + logpxobsgivenz_prev_all,0))
			#m_loglikelihood.append(loglike.item())
			m_nelbo.append(neg_bound.item())
			xm_logits = all_logits_obs_model_prev.reshape([batch_size,channels,p,q])

		imputation = iota_x
		#if with_labels:
		#	imputation = iota_x[0,0,:,:].reshape(batch_size,1,p,q)

		if data=='mnist':
			if with_labels:
				imputation[0,0,:,:].reshape(batch_size,1,p,q)[~mask] = torch.sigmoid(v)
			else:
				imputation[~mask] = torch.sigmoid(v)
			loss, loglike = mvae_loss(iota_x = imputation,mask = mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1, with_labels= with_labels, labels=labels)
		else:
			imputation[~mask] = v
			loss, loglike = mvae_loss_svhn(iota_x = imputation,mask = mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)

		## Calculate log-likelihood of the imputation

		m_loglikelihood.append(loglike.item())

		if with_labels:
			imputation = imputation[0,0,:,:].reshape(batch_size,1,p,q)

		imputation = imputation.cpu().data.numpy().reshape(channels,p,q)
		err = np.array([mse(imputation.reshape([1,channels,p,q]),full.cpu().data.numpy(),mask.cpu().data.numpy())])
		m_error.append(err)
		imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
		#print(torch.sum(logpxobsgivenz), torch.sum(logpz), torch.sum(logq))

		#xms = xgivenz.sample().reshape([L,batch_size,28*28])
		#xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
		#xm = torch.sigmoid(all_logits_obs_model).reshape([1,batch_size,28*28])
		if do_plot : 
			if (t)%interval==0 or t ==0:
				if data=='mnist':
					plot_image(np.squeeze(imputation), prefix + str(t) + "mwg.png")
				else:
					#plot_image_svhn(np.squeeze(imputation), prefix + str(t) + "mwg.png")
					print(np.max(imputation[~mask]), np.min(imputation[~mask]) )
	if do_plot:
		if data=='mnist':
			plot_image(np.squeeze(imputation), prefix + str(T) + "mwg.png")
		else:
			plot_image_svhn(np.squeeze(imputation), prefix + str(T) + "mwg.png")

		plot_images_in_row(T, loc1 =  prefix +str(0) + "mwg.png", loc2 =  prefix + str(interval) + "mwg.png", loc3 =  prefix +str(2*interval) + "mwg.png", loc4 =  prefix +str(3*interval) + "mwg.png", loc5 =  prefix +str(4*interval) + "mwg.png", file =  prefix + "mwg-all.png", data = data) 

	##Evaluate --
	do_plot=False
	if do_plot:
		if with_labels:
			iota_x[0,0,:,:].reshape(batch_size,1,p,q)[~mask] = x_prev
		else:
			iota_x[~mask] = x_prev
		iwae_bound = eval_iwae_bound(iota_x, full, mask, encoder ,decoder, p_z, d, K=1000, with_labels=with_labels, data=data)
		print("IWAE bound for metropolis within gibbs -- ", iwae_bound)

	iwae_bound =0
	if evaluate :
		return zgivenx_evaluate, logqz_evaluate
	else:
		return m_nelbo, m_error, xm_logits, m_loglikelihood, iwae_bound, x_prev


def mse_(xhat,xtrue): # MSE function for imputations
    xhat = np.array(xhat*0.5 + 0.5)
    xtrue = np.array(xtrue*0.5 + 0.5)
    num_missing = len(xhat)
    print(xhat, xtrue)
    print(np.mean(np.power(xhat-xtrue,2)))
    exit()
    return np.mean(np.power(xhat-xtrue,2))/num_missing

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
		sigma_decoder = decoder.get_parameter("log_sigma")
		#all_logits_obs_model = torch.clip(all_logits_obs_model, min=0.01, max = 0.99)        
		#all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale = (1e-3 + torch.nn.Softplus()(sigma_decoder))*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
		all_log_pxgivenz_flat = DiscNormal(loc = all_logits_obs_model.reshape([-1,1]), scale = (torch.nn.Softplus()(sigma_decoder))*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)

	#imputation_miss = all_logits_obs_model.reshape([K_samples,channels,p,q])[~b_mask]
	#img = imputation.cpu().data.numpy()
    #err = np.array([mse(img.reshape([1,-1]),full_[~b_mask].cpu().data.numpy().reshape([1,-1]))])

	#print(all_log_pxgivenz_flat)

	all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K_samples,channels*p*q])
	logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K_samples,1]) 
	logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K_samples,1]) 

	return logpxmissgivenz, all_logits_obs_model

def pass_decoder_uci(K_samples, zgivenx, decoder, b_full, b_mask, p, d, data):
	zgivenx_flat = zgivenx.reshape([K_samples,d])
	out_decoder = decoder.forward(zgivenx_flat)
    
	all_means_obs_model = out_decoder[..., :p]
	all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:]) + 0.001

	full_ = torch.Tensor.repeat(b_full,[K_samples,1]) 
	data_flat = full_.reshape([-1,1])

	mask_ = torch.Tensor.repeat(b_mask,[K_samples,1])  
	tiledmask = mask_.reshape([K_samples,p]).cuda()

	mask_ = torch.Tensor.repeat(b_mask,[K_samples,1])  
    
	all_log_pxgivenz_flat = td.Normal(loc = all_means_obs_model.reshape([-1,1]), scale = all_scales_obs_model.cuda().reshape([-1,1])).log_prob(data_flat)
    
	all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K_samples,p])
	logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K_samples,1]) 
	logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K_samples,1]) 

	return logpxmissgivenz, all_means_obs_model


def evaluate_z(p_z, b_data, b_full, b_mask, z_params, decoder, device, d, results, nb , K_samples, data='mnist', ismixture=False, to_plot=False,do_random=False):
	if ismixture:
		#print("in mixture --")
		[logits, means, scales] = z_params
		logits.requires_grad = False
		means.requires_grad = False
		scales.requires_grad = False        
		if data=='mnist':        
			q_z = ReparameterizedNormalMixture1d(logits.to(device,dtype = torch.float), means.to(device,dtype = torch.float), torch.nn.Softplus()(scales.to(device,dtype = torch.float))) 
		elif data=='svhn':
			q_z = ReparameterizedNormalMixture1d(logits.to(device,dtype = torch.float), means.to(device,dtype = torch.float), 1e-2 +  torch.nn.Softplus()(scales.to(device,dtype = torch.float)))
		else:
			q_z = ReparameterizedNormalMixture1d(logits.to(device,dtype = torch.float), means.to(device,dtype = torch.float), torch.nn.Softplus()(scales.to(device,dtype = torch.float))) 
	else:
		print("in gaussian --")
		z_params.requires_grad = False
		if data=='mnist':      
			q_z = td.Independent(td.Normal(loc=z_params[...,:d], scale=torch.nn.Softplus()(z_params[...,d:])),1)
		elif data=='svhn':
			sigma_decoder = decoder.get_parameter("log_sigma")
			scales_z = 1e-2 + torch.nn.Softplus()(z_params[...,d:])
			q_z = td.Independent(td.Normal(loc=z_params[...,:d], scale=scales_z),1)
		else:
			q_z = td.Independent(td.Normal(loc=z_params[...,:d], scale=torch.nn.Softplus()(z_params[...,d:])),1)            

	zgivenx = q_z.rsample([K_samples])
	if data != "uci":
		channels = b_full.shape[1]
		p = b_full.shape[2]
		q = b_full.shape[3]
	else:
		p = b_full.shape[1]

	logqz = q_z.log_prob(zgivenx).reshape(K_samples,1)
	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 
	logpxmissgivenz = torch.zeros((K_samples,1)).cuda()
	batches = 1000   
	if data != "uci":
		for i in range(int(K_samples/batches)):
			zgivenx_batch = zgivenx[i*batches: (i+1)*batches].reshape([batches,d])
			#b_full_batch = b_full[i*batches: (i+1)*batches,:]
			#b_mask_batch = b_mask[i*batches: (i+1)*batches,:]
			logpxmissgivenz[i*batches: (i+1)*batches] = pass_decoder(batches, zgivenx_batch, decoder, b_full, b_mask, channels, p, q,d, data)[0] #, all_logits_obs_model
	else:
		logpxmissgivenz, all_logits_obs_model = pass_decoder_uci(K_samples, zgivenx, decoder, b_full, b_mask, p, d, data)
        

	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data
	del logqz, logpz, logpxmissgivenz, zgivenx
	if to_plot:
		index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
		predicted_image = b_data
		if data == 'mnist':
			predicted_image[~b_mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~b_mask])
		elif data =='svhn':
			predicted_image[~b_mask] = all_logits_obs_model.reshape(K_samples,1,channels,p,q)[index_,~b_mask]
		#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])
		if data=='mnist':
			display_images(decoder, q_z, d, os.getcwd() + "/results/mnist-False-"  + str(-1) + "/compiled/"  + 'z.png', k = 50, b_data = b_data, b_mask=b_mask)
		else:
			display_images_svhn(decoder, q_z , d, os.getcwd() + "/results/svhn/"  + str(-1) + "/compiled/" + str(nb%10) + str(do_random) + 'mixture.png', k = 50)
		return iwae , np.squeeze(predicted_image.cpu().data.numpy())
	else:
		return iwae #, np.squeeze(predicted_image.cpu().data.numpy())

def eval_baseline(K_samples, p_z, encoder, decoder, iota_x, full, mask, d, data='mnist', with_labels=False, to_plot=False, nb=0, test=False):

	if data != "uci":
		channels = full.shape[1]
		p = full.shape[2]
		q = full.shape[3]
	else:
		p = full.shape[1]
        
	out_encoder = encoder.forward(iota_x)
	if data =='mnist':
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
	elif data=='svhn':
		scales_z = 1e-2 + torch.nn.Softplus()(out_encoder[...,d:])
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale = scales_z),1)
	else:
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
        
	logpxmissgivenz = torch.zeros((K_samples,1)).cuda()
	batches = 1000   
	if data != "uci":
		for i in range(int(K_samples/batches)):
			zgivenx_batch = zgivenx_flat[i*batches: (i+1)*batches].reshape([batches,d])
			#b_full_batch = full[i*batches: (i+1)*batches].reshape([batches,channels,p,q])
			#b_mask_batch = mask[i*batches: (i+1)*batches].reshape([batches,channels,p,q])
			logpxmissgivenz[i*batches: (i+1)*batches] = pass_decoder(batches, zgivenx_batch, decoder, full, mask, channels, p, q,d, data)[0] #, all_logits_obs_model
	else:
		logpxmissgivenz, all_logits_obs_model = pass_decoder_uci(K_samples, zgivenx_flat, decoder, full, mask, p, d, data)

	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data

	if to_plot:
		index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
		predicted_image = iota_x
		if data == 'mnist':
			predicted_image[~mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~mask])
		elif data =='svhn':
			predicted_image[~mask] = all_logits_obs_model.reshape(K_samples,1,channels,p,q)[index_,~mask]
		#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])
		if data=='mnist':
			display_images(decoder, q_zgivenxobs, d, os.getcwd() + "/results/mnist-False-"  + str(-1) + "/compiled/"  + 'encoder.png', k = 50, b_data = b_data, b_mask=b_mask)
		else:
			display_images_svhn(decoder, q_zgivenxobs , d, os.getcwd() + "/results/svhn/"  + str(-1) + "/compiled/" + str(nb%10) + str(test) + 'encoder.png', k = 50)
		return iwae , np.squeeze(predicted_image.cpu().data.numpy())
	else:
		return iwae 




##change here --
def evaluate_pseudo_gibbs(K_samples, p_z, encoder, decoder, x_init, iota_x, full, mask, d, device, data='mnist', with_labels=False):
	print("in pseudo-gibbs --")
	if data != "uci":
		channels = full.shape[1]
		p = full.shape[2]
		q = full.shape[3]
	else:
		p = full.shape[1]

	iota_x[~mask] = x_init
	zgivenx = torch.zeros((K_samples,d)).to(device,dtype = torch.float)
	logpxmissgivenz = torch.zeros([K_samples,1]).to(device,dtype = torch.float)
	logqz = torch.zeros([K_samples,1]).to(device,dtype = torch.float)

	for i in range(K_samples):
		out_encoder = encoder.forward(iota_x)
		if data =='mnist' or data=='uci':
			q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
		else:
			scales_z = 1e-2 + torch.nn.Softplus()(out_encoder[...,d:])
			q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale = scales_z),1)
            
		zgivenx[i] = q_zgivenxobs.rsample([1]).reshape(1,d) 
		logqz[i] = q_zgivenxobs.log_prob(zgivenx[i]).reshape(1,1)

		out_decoder = decoder.forward(zgivenx[i])
		if data=='mnist':
			xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=out_decoder.reshape(-1)),1)
		elif data=='svhn':
			sigma_decoder = decoder.get_parameter("log_sigma")
			xgivenz = td.Normal(loc = out_decoder.reshape([-1,1]), scale = (1e-3 + torch.nn.Softplus()(sigma_decoder))*(torch.ones(*out_decoder.shape).cuda()).reshape([-1,1]))
		else:
			all_means_obs_model = out_decoder[..., :p]
			all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:]) + 0.001
			xgivenz = td.Normal(loc = all_means_obs_model.reshape([-1,1]), scale = all_scales_obs_model.cuda().reshape([-1,1]))
		if data != "uci":   
			iota_x[~mask] = xgivenz.sample().reshape(1,channels,p,q)[~mask]
		else:
			iota_x[~mask] = xgivenz.sample().reshape(1,p)[~mask]

	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 
	logpxmissgivenz = torch.zeros((K_samples,1)).cuda()
    
	batches = 1000   
	if data != "uci":
		for i in range(int(K_samples/batches)):
			zgivenx_batch = zgivenx[i*batches: (i+1)*batches].reshape([batches,d])
			#b_full_batch = b_full[i*batches: (i+1)*batches,:]
			#b_mask_batch = b_mask[i*batches: (i+1)*batches,:]
			logpxmissgivenz[i*batches: (i+1)*batches] = pass_decoder(batches, zgivenx_batch, decoder, full, mask, channels, p, q,d, data)[0] #, all_logits_obs_model
	else:
		logpxmissgivenz, all_logits_obs_model = pass_decoder_uci(K_samples, zgivenx, decoder, full, mask, p, d, data)
        
	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data

	del zgivenx, logqz, logpz, logpxmissgivenz
	#index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
	#predicted_image = iota_x
	#predicted_image[~mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~mask])
	#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])

	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())


def evaluate_metropolis_within_gibbs(K_samples, p_z, encoder, decoder, x_init, iota_x, full, mask, d, device, data='mnist', with_labels=False):
	print("Metropolis-within-gibbs --")
	if data != "uci":
		channels = full.shape[1]
		p = full.shape[2]
		q = full.shape[3]
	else:
		p = full.shape[1]

	iota_x[~mask] = x_init
    
	if data=='mnist':
		zgivenx, logqz  = m_g_sampler(iota_x, full, mask, encoder, decoder, p_z, d, results=os.getcwd() + "/results/mnist-False-", nb=0, iterations=0, K=1, T=K_samples, data='mnist', evaluate=True)
	elif data=='svhn':
		zgivenx, logqz  = m_g_sampler(iota_x, full, mask, encoder, decoder, p_z, d, results=os.getcwd() + "/results/svhn/" , nb=0, iterations=0, K=1, T=K_samples, data=data, evaluate=True)
	else:
		zgivenx, logqz  = m_g_sampler_table(iota_x, full, mask, encoder, decoder, p_z, d, results=os.getcwd() + "/results/uci_datasets/", nb=0, iterations=0, K=1, T=K_samples,  evaluate=True)
        
        
	#logqz = q_zgivenxobs.log_prob(zgivenx).reshape(K_samples,1)
	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 
	logpxmissgivenz = torch.zeros((K_samples,1)).cuda()
	batches = 1000   
	if data != "uci":
		for i in range(int(K_samples/batches)):
			zgivenx_batch = zgivenx[i*batches: (i+1)*batches].reshape([batches,d])
			#b_full_batch = b_full[i*batches: (i+1)*batches,:]
			#b_mask_batch = b_mask[i*batches: (i+1)*batches,:]
			logpxmissgivenz[i*batches: (i+1)*batches] = pass_decoder(batches, zgivenx_batch, decoder, full, mask, channels, p, q,d, data)[0] #, all_logits_obs_model
	else:
		logpxmissgivenz, all_logits_obs_model = pass_decoder_uci(K_samples, zgivenx, decoder, full, mask, p, d, data)

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
	del logqz, logpz, logpxmissgivenz
	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())


def evaluate_iaf(p_z, b_data, b_full, b_mask, iaf_params, encoder, decoder, device, d, results, nb , K_samples, data='mnist'):
	print("in IAF --")
	if data=='mnist' :
		[t1, t2] = iaf_params
	elif data == 'uci':
		[t1, t2] = iaf_params #z_params
	else:
		[t1, t2, z_params] = iaf_params

	if data != "uci":
		autoregressive_nn =  AutoRegressiveNN(d, [320, 320]).cuda().to(device,dtype = torch.float)
		autoregressive_nn2 =  AutoRegressiveNN(d, [320, 320]).cuda().to(device,dtype = torch.float)
	else:
		autoregressive_nn =  AutoRegressiveNN(d, [8, 8]).cuda().to(device,dtype = torch.float)
		autoregressive_nn2 =  AutoRegressiveNN(d, [8, 8]).cuda().to(device,dtype = torch.float)        
        
	autoregressive_nn.load_state_dict(t1)
	autoregressive_nn2.load_state_dict(t2)

	for params in autoregressive_nn.parameters():
		params.requires_grad = False

	for params in autoregressive_nn2.parameters():
		params.requires_grad = False

	#if data=='svhn' : #or data=='uci'
	#	z_params.requires_grad = False
	#	z_params = z_params.to(device,dtype = torch.float)

	if data=='mnist':
		z_params = encoder.forward(b_data.to(device,dtype = torch.float))
		q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d], scale=torch.nn.Softplus()(z_params[...,d:])),1)
	elif data == 'svhn':
		#z_params = encoder.forward(b_data.to(device,dtype = torch.float))
		scales_z = 1e-2 + torch.nn.Softplus()(z_params[...,d:])
		q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d],scale = scales_z),1)
	else:
		z_params = encoder.forward(b_data.to(device,dtype = torch.float))
		q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d], scale=torch.nn.Softplus()(z_params[...,d:])),1)
  
	transform = AffineAutoregressive(autoregressive_nn).cuda()
	transform2 = AffineAutoregressive(autoregressive_nn2).cuda()
	pyro.module("my_transform", transform)  
	flow_dist = pyro.distributions.torch.TransformedDistribution(q_zgivenxobs, [transform, transform2]) #, transform2, transform3, transform4, transform5

	zgivenx = flow_dist.rsample([K_samples])
	if data != "uci":
		channels = b_full.shape[1]
		p = b_full.shape[2]
		q = b_full.shape[3]
	else:
		p = b_full.shape[1]

	logqz = flow_dist.log_prob(zgivenx).reshape(K_samples,1)
	logpz = p_z.log_prob(zgivenx).reshape(K_samples,1) 
	logpxmissgivenz = torch.zeros((K_samples,1)).cuda()
	batches = 1000   
	if data != "uci":
		for i in range(int(K_samples/batches)):
			zgivenx_batch = zgivenx[i*batches: (i+1)*batches].reshape([batches,d])
			#b_full_batch = b_full[i*batches: (i+1)*batches,:]
			#b_mask_batch = b_mask[i*batches: (i+1)*batches,:]
			logpxmissgivenz[i*batches: (i+1)*batches] = pass_decoder(batches, zgivenx_batch, decoder, b_full, b_mask, channels, p, q,d, data)[0] #, all_logits_obs_model
	else:
		logpxmissgivenz, all_logits_obs_model = pass_decoder_uci(K_samples, zgivenx, decoder, b_full, b_mask, p, d, data)

	iwae = np.zeros(K_samples)
	for i in np.arange(K_samples):
		iwae[i] = torch.logsumexp(logpxmissgivenz[:i] + logpz[:i] - logqz[:i], 0).cpu().data

	del logqz, logpz, logpxmissgivenz
	#index_ = torch.argmax(logpxmissgivenz + logpz - logqz)
	#predicted_image = b_data
	#predicted_image[~b_mask] = torch.sigmoid(all_logits_obs_model.reshape(K_samples,1,1,28,28)[index_,~b_mask])
	#print(logpxmissgivenz[index_] + logpz[index_] - logqz[index_])
	return iwae #, np.squeeze(predicted_image.cpu().data.numpy())


