import torch
import numpy as np
from loss import *
import gc
from plot import *
import os
from pyro.nn import AutoRegressiveNN
import pyro
from mixture import *
from init_methods import *
from sklearn.manifold import TSNE
from data import *
import matplotlib.pyplot as plt

def eval_iwae_bound_table(iota_x, full, mask, encoder ,decoder, p_z, d, K=1, with_labels=False, labels= None, data='mnist' ):
	p = iota_x.shape[1]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	out_encoder = encoder(iota_x)
	q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
	zgivenx = q_zgivenxobs.rsample([K]).reshape(K,d) 
	zgivenx_flat = zgivenx.reshape([K,d])

	logpz = p_z.log_prob(zgivenx).reshape([K,1])
	logqz = q_zgivenxobs.log_prob(zgivenx).reshape([K,1])

	out_decoder = decoder(zgivenx_flat)

	all_means_obs_model = out_decoder[..., :p]
	all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:]) + 0.001

	data_flat = torch.Tensor.repeat(full,[K,1]).reshape([-1,1])
	tiledmask = torch.Tensor.repeat(mask,[K,1])

	all_log_pxgivenz_flat = td.Normal(loc = all_means_obs_model.reshape([-1,1]), scale = all_scales_obs_model.cuda().reshape([-1,1])).log_prob(data_flat)
	all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*1,p])

	logpmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,1])
	logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,1])

	iwae_bound = torch.logsumexp(logpmissgivenz + logpz - logqz, 0) ##Plots for this.
	print(torch.mean(logpmissgivenz), torch.mean(logqz - logpz))

	return iwae_bound


def pseudo_gibbs_table(sampled_image, b_data, b_mask, encoder, decoder, p_z,  d, results, iterations, T=100, nb=0, K=1, data='mnist', full = None, evaluate=False,with_labels=False, labels = None):
	batch_size = b_data.shape[0]
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#b_mask = torch.Tensor.repeat(b_mask,[K,1,1,1])         
	i = -1
	prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 
	interval = int(T/4)

	do_plot = False
	#print(torch.cuda.current_device())

	for l in range(T):
		out_encoder = encoder.forward(sampled_image)
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
		zgivenx = q_zgivenxobs.rsample([K])

		if with_labels:
			labels = torch.zeros(1,labels.shape[0],10).to(device,dtype = torch.float)
			zgivenx_y = torch.cat((zgivenx,labels),2)
			#print(zgivenx_y.shape)
			zgivenx_flat = zgivenx_y.reshape([1,d+10])
		else:
			zgivenx_flat = zgivenx.reshape([K*batch_size,d])

		x_logits = decoder.forward(zgivenx_flat)

		if data=='mnist':
			#a = torch.sigmoid(x_logits)
			#a = torch.mean(a)
			xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits.reshape(-1)),1)
			#xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(probs = a.reshape(-1)),1)
		else:
			sigma_decoder = decoder.get_parameter("log_sigma")
			xgivenz = td.Normal(loc = x_logits.reshape([-1,1]), scale =  sigma_decoder.exp()*(torch.ones(*x_logits.shape).cuda()).reshape([-1,1]))

		#all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,28*28])
		#logpxobsgivenz = torch.sum(all_log_pxgivenz,1).reshape([K,batch_size])
		#xms = xgivenz.sample().reshape([L,batch_size,p])		
		#sampled_image[~b_mask] = xgivenz.sample().reshape(K*batch_size,channels,p,q)[~b_mask]
		if with_labels:
			sample_i = xgivenz.sample().reshape(K*batch_size,1,p,q)[~b_mask]
			sampled_image[0,0,:,:].reshape([1,1,28,28])[~b_mask] = sample_i
		else:
			sample_i = xgivenz.sample().reshape(K*batch_size,channels,p,q)[~b_mask]
			sampled_image[~b_mask] = sample_i

		if do_plot:
			if (l)%interval==0 or l==0:
				#print(x_logits.shape)
				if data=='mnist':
					a = torch.sigmoid(x_logits)
					plot_image(np.squeeze(a.cpu().data.numpy()), prefix + str(l) + "pg.png")
				else:
					a = x_logits
					##SVHN prefix
					plot_image_svhn(np.squeeze(a.cpu().data.numpy()),prefix + str(l) + "pg.png")

	if do_plot:
		if data=='mnist':
			a = torch.sigmoid(x_logits)
			plot_image(np.squeeze(a.cpu().data.numpy()),prefix + str(T) + "pg.png")
		else:
			a = x_logits
			##SVHN prefix
			plot_image_svhn(np.squeeze(a.cpu().data.numpy()),prefix + str(T) + "pg.png")

		plot_images_in_row(T, loc1 =  prefix +str(0) + "pg.png", loc2 =  prefix +str(interval) + "pg.png", loc3 =  prefix +str(2*interval) + "pg.png", loc4 =  prefix +str(3*interval) + "pg.png", loc5 =  prefix +str(4*interval) + "pg.png", file =  prefix + "pg-all.png", data = data) 

	##Evaluate --	
	do_plot=True
	if do_plot:
		iwae_bound = eval_iwae_bound(sampled_image, full, b_mask, encoder ,decoder, p_z, d, K=1000, with_labels = with_labels, data= data)
		print("IWAE bound for pseudo-gibbs -- ", iwae_bound)

	iwae_bound=0

	if data=='mnist':
		return x_logits.reshape(1,1,p,q), sampled_image, iwae_bound, sample_i
	else:
		return x_logits.reshape(1,channels,p,q), sampled_image, iwae_bound, sample_i


def m_g_sampler_table(iota_x, full, mask, encoder, decoder, p_z, d, results, nb, iterations, K=1, T=1000, data='mnist', evaluate=False, with_labels=False, labels= None):
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
	q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
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

		if with_labels:
			iota_x[0,0,:,:].reshape([1,1,28,28])[~mask] = x_prev
		else:
			iota_x[~mask] = x_prev

		out_encoder = encoder.forward(iota_x)
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

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
			all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale =  sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
			all_log_pxgivenz_flat_prev = td.Normal(loc = all_logits_obs_model_prev.reshape([-1,1]), scale =  sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)

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
			xgivenz = td.Normal(loc = all_logits_obs_model, scale =  sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()))

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
				xgivenz_prev = td.Normal(loc = all_logits_obs_model_prev, scale =  sigma_decoder.exp()*(torch.ones(*all_logits_obs_model_prev.shape).cuda()))

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
					plot_image_svhn(np.squeeze(imputation), prefix + str(t) + "mwg.png")

	if do_plot:
		if data=='mnist':
			plot_image(np.squeeze(imputation), prefix + str(T) + "mwg.png")
		else:
			plot_image_svhn(np.squeeze(imputation), prefix + str(T) + "mwg.png")

		plot_images_in_row(T, loc1 =  prefix +str(0) + "mwg.png", loc2 =  prefix + str(interval) + "mwg.png", loc3 =  prefix +str(2*interval) + "mwg.png", loc4 =  prefix +str(3*interval) + "mwg.png", loc5 =  prefix +str(4*interval) + "mwg.png", file =  prefix + "mwg-all.png", data = data) 

	##Evaluate --
	do_plot=True
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



def optimize_q_xm_table(num_epochs, xm_params, z_params, b_data, sampled_image_o, b_mask, b_full, encoder, decoder, device, d , results, iterations, nb, file, p_z , K_samples, data='mnist', scales = None, p_z_eval=None):
	beta_0 = 1
	xm_params = xm_params.to(device)
	xm_params = torch.clamp(xm_params, min=-10, max=10)
	#print("Image: ", nb, " iter: ", iterations, xm_params)

	batch_size = b_data.shape[0]
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	xm_params.requires_grad = True

	if data == 'svhn':
		scales.requires_grad = True

	if data =='mnist':
		test_optimizer = torch.optim.Adam([xm_params], lr=1.0, betas=(0.9, 0.999))  #0.1
	else:
		test_optimizer = torch.optim.Adam([xm_params, scales], lr=1.0, betas=(0.9, 0.999))             

	scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=30, gamma=0.1)

	xm_elbo = np.zeros((num_epochs))
	xm_mse = np.zeros((num_epochs))
	xm_loglikelihood = np.zeros((num_epochs))
	xm_loss= np.zeros((num_epochs))

	term1 = np.zeros((6, 10, num_epochs))  #loglikelihood
	term2 = np.zeros((6, 10, num_epochs))  #KL
	term3 = np.zeros((6, 10, num_epochs))  #Entropy
	i = -1

	prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 

	a = int(num_epochs/4)
	for k in range(num_epochs):
		#print(xm_params, z_params)

		beta = beta_0
		test_optimizer.zero_grad()
		loss, log_like, aa, bb, cc = xm_loss_q(iota_x = b_data.to(device,dtype = torch.float), sampled = sampled_image_o.to(device,dtype = torch.float), mask = b_mask,p_z = p_z, z_params = z_params, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta=beta, K=K_samples, K_z= 1, epoch=k, data=data, scales=scales)
		loss.backward()            
		test_optimizer.step()
		scheduler.step()

		##Calculate loglikelihood and mse on imputation
		imputation = b_data.to(device,dtype = torch.float)

		if data=='mnist':
			v = torch.sigmoid(xm_params).detach()
			imputation[~b_mask] = v
		else:
			imputation[~b_mask] = xm_params.detach().to(device,dtype = torch.float)

		loss, log_like, aa, bb, cc = xm_loss_q(iota_x = imputation.to(device,dtype = torch.float), sampled = imputation.to(device,dtype = torch.float), mask = b_mask,p_z = p_z, z_params = z_params, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta=beta, K=K_samples, K_z= 1, train=False, epoch=k, data=data, scales=scales)
		xm_loglikelihood[k] += log_like.item()

		### Get mse error on imputation
		imputation = imputation.cpu().data.numpy().reshape(channels, p, q)
		err = np.array([mse(imputation.reshape([1,channels,p,q]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
		xm_mse[k] += err
		loss, log_like, aa, bb, cc = xm_loss_q(iota_x = b_data.to(device,dtype = torch.float), sampled = sampled_image_o.to(device,dtype = torch.float), mask = b_mask,p_z = p_z, z_params = z_params, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta=beta, K=K_samples, K_z= 1, epoch=k, data=data, scales=scales)

		#term1[iterations, nb%10, k] = aa.item()
		#term2[iterations, nb%10, k] = bb.item()
		#term3[iterations, nb%10, k] = cc.item()
		#print(aa.item(), bb.item(), cc.item())
		xm_loss[k] += loss.item()
		xm_elbo[k] += -loss.item()
		#xm_loss_per_img[iterations, nb%10, k] = loss.item() 
		#xm_mse_per_img[iterations, nb%10, k] = err

		#if(k==num_epochs-1):
		#print(log_like.item())
		#print("Loss with our method: ", loss.item())

		if (k)%a==0 or k ==0:
			if data=='mnist':
				plot_image(np.squeeze(imputation),prefix + str(k) + "q_xm.png")
			else:
				plot_image_svhn(np.squeeze(imputation), prefix + str(k) + "q_xm.png")

	iwae_loss, log_like, aa, bb, cc = xm_loss_q(iota_x = b_data.to(device,dtype = torch.float), sampled = sampled_image_o.to(device,dtype = torch.float), mask = b_mask, p_z = p_z_eval, z_params = z_params, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta=beta, K=K_samples, K_z= 1, epoch=k, data=data, scales=scales, evaluate=True, full= b_full.to(device,dtype = torch.float))

	print("IWAE for q_xm", -iwae_loss)

	do_plot = False
	if do_plot:
		if data=='mnist':
			plot_image(np.squeeze(imputation), prefix + str(num_epochs) + "q_xm.png")
		else:
			plot_image_svhn(np.squeeze(imputation), prefix + str(num_epochs) + "q_xm.png")

		if data=='mnist':
			p_xm = td.Independent(td.ContinuousBernoulli(logits=(xm_params).cuda()),1)       ##changed to probs, since sigmoid outputs  
			display_images_from_distribution(b_data.cpu(), b_mask.cpu(), p_xm, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '-samples-' + file , k = 50, data='mnist')

		else:
			#epsilon = 0.01*torch.ones(*xm_params.shape).cuda()
			p_xm = td.Independent(td.Normal(loc=xm_params.cuda(),scale=scales.exp()),1)
			display_images_from_distribution(b_data.cpu(), b_mask.cpu(), p_xm, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '-samples-' + file , k = 50, data='svhn')


		plot_images_in_row(num_epochs, loc1 =  prefix +str(0) + "q_xm.png", loc2 =  prefix +str(a) + "q_xm.png", loc3 =  prefix +str(2*a) + "q_xm.png", loc4 =  prefix +str(3*a) + "q_xm.png", loc5 =  prefix +str(4*a) + "q_xm.png", file =  prefix + file, data=data) 

	return xm_loss, xm_mse, -iwae_loss


def optimize_mixture_IAF_table(num_epochs, p_z, logits, means, scales, b_data, b_full, b_mask , encoder, decoder, device, d, results, iterations, nb, K_samples, data='mnist', with_labels=False, labels = None, do_random=True):
	i=-1
	batch_size = b_data.shape[0]
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	num_components = 10

	b_data[~b_mask] = 0
	r1 = -1
	r2 = 1
	#print(b_data.shape, type(b_data)) 
	do_plot = True
	logits = logits.to(device,dtype = torch.float)
	means = means.to(device,dtype = torch.float)
	scales = scales.to(device,dtype = torch.float)

	autoregressive_nn =  AutoRegressiveNN(d, [320, 320]).cuda()
	autoregressive_nn2 =  AutoRegressiveNN(d, [320, 320]).cuda()

	sd = autoregressive_nn.state_dict()
	sd['layers.2.weight'] = sd['layers.2.weight']*0.0001
	sd['layers.2.bias'] = sd['layers.2.bias']*0.0001
	autoregressive_nn.load_state_dict(sd)

	sd = autoregressive_nn2.state_dict()
	sd['layers.2.weight'] = sd['layers.2.weight']*0.0001
	sd['layers.2.bias'] = sd['layers.2.bias']*0.0001
	autoregressive_nn2.load_state_dict(sd)

	if do_plot:
		comp_samples = np.zeros((num_components+2,50,d))
		for comp in range(num_components):
			probs = torch.zeros(batch_size, num_components)
			probs[0,comp] = 1.0
			dist = td.MixtureSameFamily(td.Categorical(probs=probs.to(device,dtype = torch.float)), td.Independent(td.Normal(means, scales.exp()), 1))
			transform = AffineAutoregressive(autoregressive_nn).cuda()
			transform2 = AffineAutoregressive(autoregressive_nn2).cuda()

			pyro.module("my_transform", transform)  
			flow_dist_ = pyro.distributions.torch.TransformedDistribution(dist, [transform, transform2])
			#print(dist.sample([50]).cpu().data.numpy(), dist.sample([50]).cpu().data.numpy().shape)
			a = flow_dist_.sample([50])
			comp_samples[comp] = a.cpu().data.numpy().reshape([50,d]) 
			display_images(decoder, flow_dist_, d, results + str(i) + "/compiled/components/" + str(nb%10)  + str(iterations)  + '-component' + str(comp) + 'init-mixture.png', k = 50, data=data)

		if data == 'mnist':
			x_7 = get_sample_digit(data_dir = "data" , digit=7, file = results + str(i) + "/compiled/")
			x_9 = get_sample_digit(data_dir = "data" , digit=9 ,file = results + str(i) + "/compiled/")
			#x_7_embed = = encoder.forward(x_7.to(device,dtype = torch.float))
			#x_
			out_7 = encoder.forward(x_7.to(device,dtype = torch.float))
			out_9 = encoder.forward(x_9.to(device,dtype = torch.float))
			comp_samples[num_components] =  out_7[...,:d].cpu().data.numpy().reshape([50,d]) 
			comp_samples[num_components+1] = out_9[...,:d].cpu().data.numpy().reshape([50,d]) 
			print(get_pxgivenz(out_7,d, x_7.to(device,dtype = torch.float),decoder), get_pxgivenz(out_9,d, x_9.to(device,dtype = torch.float), decoder))

		comp_samples = comp_samples.reshape(((num_components+2)*50,d))

		X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(comp_samples)
		scatter_plot_(X.reshape((num_components+2,50,2)), num_components, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'init-mixture-scatterplot.png')

		print("entropy before------", get_entropy_mixture(logits))
		print(td.Independent(td.Normal(means, scales.exp()), 1).entropy())

	#logits.requires_grad = True
	#means.requires_grad = True
	#scales.requires_grad = True

	#test_optimizer = torch.optim.Adam([means,scales], lr=1.0, betas=(0.9, 0.999)) 
	#test_optimizer_logits = torch.optim.Adam([logits], lr=0.1, betas=(0.9, 0.999))  #1.0 for re-inits
	optimizer_iaf = torch.optim.Adam(list(autoregressive_nn.parameters()) + list(autoregressive_nn2.parameters()), lr=0.01)

	#lambda1 = lambda epoch: 0.67 ** epoch
	lambda1 = lambda epoch: 0.95
	#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers, lr_lambda=lambda1)
	#scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=30, gamma=0.5)
	#scheduler_logits = torch.optim.lr_scheduler.StepLR(test_optimizer_logits, step_size=30, gamma=0.5)
	#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(test_optimizer, lr_lambda=lambda1)
	#scheduler_logits = torch.optim.lr_scheduler.MultiplicativeLR(test_optimizer_logits, lr_lambda=lambda1)

	lrs = []

	#test_optimizer = torch.optim.Adagrad([xm_params], lr=0.1) 
	#b_data = b_data_init
	#q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d],scale=torch.nn.Softplus()(z_params[...,d:])),1)

	z_elbo = np.zeros((num_epochs))
	z_mse = np.zeros((num_epochs))
	z_loglikelihood = np.zeros((num_epochs))
	z_loss= np.zeros((num_epochs))

	a = int(num_epochs/4)
	beta = 1

	#do_random = True
	do_plot = True

	for k in range(num_epochs):
		##Let the optimization find modes first
		#if not do_random and k==0:
		#	logits.requires_grad = False

		#if do_random and k==99:
		#	logits.requires_grad = False

		##Adjust the weights of the modes in the mixture
		#if k==199:
		#	logits.requires_grad = True

		##Reset learning rate for means and scales
		#if do_random and k==99:
		#	del test_optimizer, scheduler
		#	test_optimizer = torch.optim.Adam([means,scales], lr=1.0, betas=(0.9, 0.999)) 
		#	scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=30, gamma=0.5)

		#test_optimizer.zero_grad()
		optimizer_iaf.zero_grad()
		#test_optimizer_logits.zero_grad()
		loss, log_like, aa, bb, flow_dist = mixture_loss(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, means = means, scales = scales, logits = logits,  decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta, iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2)
		loss.backward() 
		#test_optimizer.step()
		#test_optimizer_logits.step()

		#if not do_random and k>=199:
		optimizer_iaf.step()

		#if do_random and k>=199:
		#	optimizer_iaf.step()

		#lrs.append(test_optimizer.param_groups[0]["lr"])
		#scheduler.step()

		#if k>=199:
		#	scheduler_logits.step()

		#loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, xm_params = xm_params.to(device,dtype = torch.float), encoder = encoder, decoder = decoder , device = device, d=d, K=100, K_z=1)

		z_loss[k] = loss.item()
		#print("k :", k, loss.item())
		#Impute 
		if do_plot:
			with torch.no_grad():
				imputation = b_data.to(device,dtype = torch.float) 
				q_z = ReparameterizedNormalMixture1d(logits.detach(), means.detach(), torch.nn.Softplus()(scales).detach())
				zgivenx = q_z.sample([])
				zgivenx_flat = zgivenx.reshape([1,d])
				all_logits_obs_model = decoder.forward(zgivenx_flat)
				if data=='mnist':
					imputation[~b_mask] = torch.sigmoid(all_logits_obs_model)[~b_mask]
				else:
					imputation[~b_mask] = all_logits_obs_model[~b_mask]

				#G/et error on imputation
				img = imputation.cpu().data.numpy()
				err = np.array([mse(img.reshape([1,channels,p,q]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
				z_mse[k] = err

				if (k)%a==0 or k ==0:
					if data=='mnist':
						plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "mixture.png")
					else:
						imputation[~b_mask] = all_logits_obs_model[~b_mask]
						#print(all_logits_obs_model[~b_mask])
						img = imputation.cpu().data.numpy()
						plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "mixture.png")

		ap = False
		if do_random and k%20==0 and k<=100 and ap: #
			#print("k ---", k )
			threshold = 0.01
			probs = torch.softmax(logits.detach(),dim=1)
			#print(probs)
			ap = torch.any(probs<threshold)
			#print(ap)
			if ap:
				b_data[~b_mask] = 0
				logits_, means_, scales_ = init_mixture(encoder, decoder, p_z, b_data, b_mask, num_components, batch_size, d, r1, r2, data=data, repeat=True)
				logits_ = logits_.to(device,dtype = torch.float)
				means_ = means_.to(device,dtype = torch.float)
				scales_ = scales_.to(device,dtype = torch.float)
				#logits = torch.where(probs > threshold, logits.detach(), logits_)

		#For the first 100 iterations, re-init means 
		if do_random and k%20==0 and k<=100 and ap:
			means__ = means.detach()
			scales__ = scales.detach()
			logits = logits_
			probs = torch.Tensor.repeat(probs.reshape(batch_size, num_components, 1),[1,1,d]) 
			means = torch.where(probs >= threshold, means__, means_)
			scales = torch.where(probs >= threshold, scales__, scales_)
			#print(logits, logits_)
			logits.requires_grad = True
			means.requires_grad = True
			scales.requires_grad = True

			lr_ = test_optimizer.param_groups[0]["lr"]
			del test_optimizer
			test_optimizer = torch.optim.Adam([means,scales], lr=lr_, betas=(0.9, 0.999)) ##Ask Gabe about any alternative

			lr_ = test_optimizer_logits.param_groups[0]["lr"]
			del test_optimizer_logits 
			test_optimizer_logits = torch.optim.Adam([logits], lr=0.1, betas=(0.9, 0.999)) ##Ask Gabe about any alternative

	#logits.requires_grad = False
	#means.requires_grad = False
	#scales.requires_grad = False

	for params in autoregressive_nn.parameters():
		params.requires_grad = False

	for params in autoregressive_nn2.parameters():
		params.requires_grad = False

	iwae_loss = 0
	iwae_loss, log_like, aa, bb, flow_dist = mixture_loss(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, means = means, scales = scales, logits = logits,  decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta, evaluate=True, full= b_full.to(device,dtype = torch.float), iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2)

	if do_plot:
		iwae_loss, log_like, aa, bb, flow_dist = mixture_loss(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, means = means, scales = scales, logits = logits,  decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta, evaluate=True, full= b_full.to(device,dtype = torch.float), iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2)


	print("IWAE Loss for mixture", -iwae_loss)
	#print("means after --", torch.sum(means,2))

	do_plot=True

	if do_plot:
		comp_samples = np.zeros((num_components+2,50,d))
		for comp in range(num_components):
			probs = torch.zeros(batch_size, num_components)
			probs[0,comp] = 1.0
			dist = td.MixtureSameFamily(td.Categorical(probs=probs.to(device,dtype = torch.float)), td.Independent(td.Normal(means.detach(), scales.exp().detach()), 1))
			transform = AffineAutoregressive(autoregressive_nn).cuda()
			transform2 = AffineAutoregressive(autoregressive_nn2).cuda()

			pyro.module("my_transform", transform)  
			flow_dist_ = pyro.distributions.torch.TransformedDistribution(q_z, [transform, transform2])
			#print(dist.sample([50]).cpu().data.numpy(), dist.sample([50]).cpu().data.numpy().shape)
			comp_samples[comp] = flow_dist_.sample([50]).cpu().data.numpy().reshape([50,d])
			display_images(decoder, flow_dist_, d, results + str(i) + "/compiled/components/" + str(nb%10)  + str(iterations)  + '-component' + str(comp) + 'mixture.png', k = 50, data=data)
		
		if data== 'mnist':
			comp_samples[num_components] =  out_7[...,:d].cpu().data.numpy().reshape([50,d]) 
			comp_samples[num_components+1] = out_9[...,:d].cpu().data.numpy().reshape([50,d]) 

		comp_samples = comp_samples.reshape([(num_components+2)*50,d])
		X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(comp_samples)
		scatter_plot_(X.reshape([num_components+2,50,2]), num_components, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'mixture-scatterplot.png')

		print("entropy after------", get_entropy_mixture(logits))
		print(td.Independent(td.Normal(means, scales.exp()), 1).entropy())
		#print(td.MixtureSameFamily(td.Categorical(logits=logits), td.Independent(td.Normal(means.detach(), scales.exp().detach()), 1)).entropy())
		#print(scales.exp())
		if data=='mnist':
			plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "mixture.png")
		else:
			imputation = b_data.to(device,dtype = torch.float) 
			imputation[~b_mask] = all_logits_obs_model[~b_mask]
			img = imputation.cpu().data.numpy()
			plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "mixture.png")

		prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 
		plot_images_in_row(num_epochs, loc1 =  prefix +str(0) + "mixture.png", loc2 =  prefix +str(a) + "mixture.png", loc3 =  prefix +str(2*a) + "mixture.png", loc4 =  prefix +str(3*a) + "mixture.png", loc5 =  prefix +str(4*a) + "mixture.png", file =  prefix + "mixture-all.png", data=data) 

	q_z = ReparameterizedNormalMixture1d(logits, means, torch.nn.Softplus()(scales))

	if data=='mnist':
		display_images(decoder, flow_dist, d, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + str(do_random) + 'mixture-IAF.png', k = 50, b_data = b_data, b_mask=b_mask)
	else:
		display_images_svhn(decoder, flow_dist, d, results  + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'mixture-IAF.png', k = 50)

	#del test_optimizer, test_optimizer_logits, scheduler, scheduler_logits
	return z_loss, z_mse, -iwae_loss,  autoregressive_nn, autoregressive_nn2


def optimize_IAF_table(num_epochs, z_params, b_data,  b_mask , b_full, p_z, encoder, decoder, device, d, results, iterations, nb, K_samples, p_z_eval = None, with_gaussian=False,  return_imputation=False):

	autoregressive_nn =  AutoRegressiveNN(d, [8, 8]).cuda()
	autoregressive_nn2 =  AutoRegressiveNN(d, [8, 8]).cuda()
	#autoregressive_nn3 =  AutoRegressiveNN(d, [5*d, 5*d, 5*d, 5*d, 5*d, 5*d, 5*d, 5*d ]).cuda()
	#autoregressive_nn4 =  AutoRegressiveNN(d, [5*d, 5*d, 5*d, 5*d, 5*d, 5*d, 5*d, 5*d ]).cuda()
	#autoregressive_nn5 =  AutoRegressiveNN(d, [5*d, 5*d, 5*d, 5*d, 5*d, 5*d, 5*d, 5*d ]).cuda()

	p = b_data.shape[1]
	z_params =  encoder.forward(b_data.to(device,dtype = torch.float))

	#print(torch.cuda.current_device())
	i=-1

	z_params = z_params.to(device)

	#print(z_params)
	if with_gaussian:
		z_params.requires_grad = True
		test_optimizer_z = torch.optim.Adam([z_params], lr=0.1, betas=(0.9, 0.999)) #0.1

	optimizer_iaf = torch.optim.Adam(list(autoregressive_nn.parameters()) + list(autoregressive_nn2.parameters()), lr=0.01)
	
	#print(autoregressive_nn.parameters())
	z_elbo = np.zeros((num_epochs))
	z_mse = np.zeros((num_epochs))
	z_loglikelihood = np.zeros((num_epochs))
	z_loss= np.zeros((num_epochs))

	i = -1
	a = int(num_epochs/4)
	prefix = results 

	do_plot = True
	for k in range(num_epochs):       
		#print("In IAF----")  

		loss, log_like, aa, bb, flow_dist = z_loss_table(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2)

		#z_kl_loss[iterations, nb%10, k] = bb.item()
		#z_loglike_loss[iterations, nb%10, k] = aa.item()
		if with_gaussian:
			test_optimizer_z.zero_grad()
		optimizer_iaf.zero_grad()
		loss.backward() 
		#print(list(autoregressive_nn.parameters()))
		#print(loss.item())
		optimizer_iaf.step()
		if with_gaussian:
			test_optimizer_z.step()
			#print(k, loss.item(), z_params)
		#scheduler.step()
		#loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, xm_params = xm_params.to(device,dtype = torch.float), encoder = encoder, decoder = decoder , device = device, d=d, K=100, K_z=1)

		z_loss[k] = loss.item()

		imputation = b_data.to(device,dtype = torch.float) 

		if do_plot:
			with torch.no_grad():
				zgivenx = flow_dist.rsample([])
				zgivenx_flat = zgivenx.reshape([1,d])
				all_logits_obs_model = decoder.forward(zgivenx_flat)

				imputation[~b_mask] = all_logits_obs_model[:,:p][~b_mask]

				img = imputation.cpu().data.numpy()
				err = np.array([mse(img.reshape([1,p]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
				z_mse[k] = err

				if (k)%a==0 or k ==0:
					print(imputation)

	for params in autoregressive_nn.parameters():
		params.requires_grad = False

	for params in autoregressive_nn2.parameters():
		params.requires_grad = False

	if with_gaussian:
		z_params.requires_grad = False

	iwae_loss = 0
	iwae_loss, log_like, aa, bb, flow_dist = z_loss_table(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z_eval, z_params = z_params,  encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1,iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2, evaluate=True,full= b_full.to(device,dtype = torch.float))

	
	print("IWAE bound for IAF", -iwae_loss)
	do_plot = False

	if do_plot:
		iwae_loss, log_like, aa, bb, flow_dist = z_loss_table(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z_eval, z_params = z_params,  encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2, evaluate=True,full= b_full.to(device,dtype = torch.float))

		print("IWAE Loss for IAF", -iwae_loss)
		if data=='mnist':
			plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "iafz.png")
		else:
			plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "iafz.png")
		
		plot_images_in_row(num_epochs, loc1 =  prefix +str(0) + "iafz.png", loc2 =  prefix +str(a) + "iafz.png", loc3 =  prefix +str(2*a) + "iafz.png", loc4 =  prefix +str(3*a) + "iafz.png", loc5 =  prefix +str(4*a) + "iafz.png", file =  prefix + "iaf-all.png", data = 'data') 

		if data=='mnist':
			display_images(decoder, flow_dist, d, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + str(with_gaussian) + 'iafz.png', k = 50)
		else:
			display_images_svhn(decoder, flow_dist, d, results  + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'iafz.png', k = 50)

	if return_imputation: 
		z_loss, z_mse,  -iwae_loss, autoregressive_nn, autoregressive_nn2, img
	elif with_gaussian:
		return z_loss, z_mse,  -iwae_loss, autoregressive_nn, autoregressive_nn2, z_params
	else:
		return z_loss, z_mse,  -iwae_loss, autoregressive_nn, autoregressive_nn2


def optimize_IAF_with_labels(num_epochs, p_z, p_y, b_data, b_full, b_mask , encoder, decoder, discriminative_model, device, d, results, iterations, nb, K_samples, data='mnist'):

	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	r1 = -1
	r2 = 1
	logits_y , means, scales  = init_params_labels(encoder, decoder, discriminative_model, p_z, b_data, b_mask, d, r1, r2, data=data)
	i=-1

	t1 = []
	t2 = []

	autoregressive_nn =  AutoRegressiveNN(d, [320, 320]).cuda()
	autoregressive_nn2 =  AutoRegressiveNN(d, [320, 320]).cuda()
	#print(autoregressive_nn.parameters()) 
	
	#iaf_params = list()
	#for i in range(10):
	#	t1.append(AutoRegressiveNN(d, [320, 320]).cuda())
	#	t2.append(AutoRegressiveNN(d, [320, 320]).cuda())
	#print(t1[i].parameters())
	#	iaf_params +=  list(t1[i].parameters()) + list(t2[i].parameters())

	#optimizer_iaf = torch.optim.Adam(iaf_params, lr=0.01) #+ list([t1[i].parameters() for i in range(10)])
	
	optimizer_iaf = torch.optim.Adam(list(autoregressive_nn.parameters()) + list(autoregressive_nn2.parameters()), lr=0.01)
	means = means.to(device)
	scales = scales.to(device)
	logits_y = logits_y.to(device)

	logits_y.requires_grad = True
	test_optimizer_logits = torch.optim.Adam([logits_y], lr=0.01, betas=(0.9, 0.999))  #1.0 for re-inits

	#lambda1 = lambda epoch: 0.67 ** epoch
	#lambda1 = lambda epoch: 0.95
	#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers, lr_lambda=lambda1)
	#scheduler_logits = torch.optim.lr_scheduler.StepLR(test_optimizer_logits, step_size=50, gamma=0.5)

	#optimizer_iaf = torch.optim.Adam(list(autoregressive_nn.parameters()) + list(autoregressive_nn2.parameters()), lr=0.01)
	#lambda1 = lambda epoch: 0.67 ** epoch
	#lambda1 = lambda epoch: 0.95

	z_elbo = np.zeros((num_epochs))
	z_mse = np.zeros((num_epochs))
	z_loglikelihood = np.zeros((num_epochs))
	z_loss= np.zeros((num_epochs))

	a = int(num_epochs/4)
	y_kl = np.zeros((num_epochs)) 
	z_loglike_loss = np.zeros((num_epochs)) 
	z_kl = np.zeros((num_epochs)) 

	for k in range(num_epochs):
		optimizer_iaf.zero_grad()
		test_optimizer_logits.zero_grad()
		loss, log_like, aa, bb, cc = z_loss_with_labels(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, p_y = p_y, means = means, scales = scales, logits_y = logits_y, encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2)
		y_kl[ k] = bb.item()
		z_loglike_loss[ k] = aa.item()
		z_kl[ k] = cc.item()

		loss.backward() 
		optimizer_iaf.step()
		test_optimizer_logits.step()

		#if k>=10:
		#scheduler_logits.step()
		#loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, xm_params = xm_params.to(device,dtype = torch.float), encoder = encoder, decoder = decoder , device = device, d=d, K=100, K_z=1)

		z_loss[k] = loss.item()
		
		#print(z_loss[k])
		##Imputations
		impute = False

		if impute:
			if (k)%a==0 or k ==0:
				if data=='mnist':
					plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "z-output.png")
				else:
					imputation[~b_mask] = all_logits_obs_model[~b_mask]
					#print(all_logits_obs_model[~b_mask])
					img = imputation.cpu().data.numpy()
					plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "z-output.png")

	for params in autoregressive_nn.parameters():
		params.requires_grad = False

	for params in autoregressive_nn2.parameters():
		params.requires_grad = False

	#for i in range(10):
	#	for params in t1[i].parameters():
	#		params.requires_grad = False
	#	for params in t2[i].parameters():
	#		params.requires_grad = False

	logits_y.requires_grad = False
	
	print("k , loss", k , loss.item())

	iwae_loss, log_like, aa, bb, cc = z_loss_with_labels(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, p_y = p_y, means = means, scales = scales, logits_y = logits_y, encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, evaluate=True, full = b_full, iaf=True, autoregressive_nn = autoregressive_nn, autoregressive_nn2= autoregressive_nn2)

	print("IWAE_bound for M2_z", -iwae_loss)
	print("q_y after ---", torch.nn.functional.softmax(logits_y, dim=1))

	if data=='mnist':
		display_images_with_labels(decoder, means, torch.nn.Softplus()(scales), logits_y, d, results + str(-1) + "/compiled/" + str(nb%10)  + str(iterations)  + 'labels.png', k = 50,b_data=b_data, b_mask = b_mask, directory2 = results + str(-1) + "/compiled/" + str(nb%10)  + str(iterations) + 'component', file2='labels.png' )
	else:
		display_images_svhn(decoder, q_zgivenxobs, d, results  + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '.png', k = 50)

	return z_loss, z_mse

def optimize_z(num_epochs, p_z, b_data, b_full, b_mask , encoder, decoder, device, d, results, iterations, nb, K_samples, data='mnist', p_z_eval=None, return_imputation=False):
	if iterations==-1:
		z_params =  encoder.forward(b_full.to(device,dtype = torch.float))
	else:
		z_params =  encoder.forward(b_data.to(device,dtype = torch.float))

	#print(torch.cuda.current_device())

	i=-1
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	z_params = z_params.to(device)
	z_params.requires_grad = True

	#test_optimizer = torch.optim.LBFGS([xm_params]) ##change, sgd, adagrad
	test_optimizer_z = torch.optim.Adam([z_params], lr=0.1, betas=(0.9, 0.999)) #0.1
	#scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer_z, step_size=30, gamma=0.1)

	#print(z_params)
	#test_optimizer = torch.optim.Adagrad([xm_params], lr=0.1) 
	#b_data = b_data_init
	q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d],scale=torch.nn.Softplus()(z_params[...,d:])),1)

	z_elbo = np.zeros((num_epochs))
	z_mse = np.zeros((num_epochs))
	z_loglikelihood = np.zeros((num_epochs))
	z_loss= np.zeros((num_epochs))

	a = int(num_epochs/4)

	do_plot = False
	for k in range(num_epochs):
		test_optimizer_z.zero_grad()
		loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data)

		#z_kl_loss[iterations, nb%10, k] = bb.item()
		#z_loglike_loss[iterations, nb%10, k] = aa.item()
		loss.backward() 
		test_optimizer_z.step()
		#scheduler.step()
		#loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, xm_params = xm_params.to(device,dtype = torch.float), encoder = encoder, decoder = decoder , device = device, d=d, K=100, K_z=1)

		z_loss[k] = loss.item()
		
		#print(z_loss[k])
		if do_plot:
			with torch.no_grad():
				imputation = b_data.to(device,dtype = torch.float) 
				v = z_params.detach()
				q_zgivenxobs = td.Independent(td.Normal(loc=v[...,:d],scale=torch.nn.Softplus()(v[...,d:])),1)
				zgivenx = q_zgivenxobs.rsample([])
				zgivenx_flat = zgivenx.reshape([1,d])
				all_logits_obs_model = decoder.forward(zgivenx_flat)

				if data=='mnist':
					imputation[~b_mask] = torch.sigmoid(all_logits_obs_model)[~b_mask]
				else:
					imputation[~b_mask] = all_logits_obs_model[~b_mask]

				img = imputation.cpu().data.numpy()
				err = np.array([mse(img.reshape([1,channels,p,q]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
				z_mse[k] = err

				if (k)%a==0 or k ==0:
					if data=='mnist':
						plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "z-output.png")
					else:
						imputation[~b_mask] = all_logits_obs_model[~b_mask]
						#print(all_logits_obs_model[~b_mask])
						img = imputation.cpu().data.numpy()
						plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "z-output.png")

	z_params.requires_grad = False
	iwae_loss=0

	do_plot=True
	if do_plot:
		iwae_loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z_eval, z_params = z_params, encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, evaluate=True, full= b_full.to(device,dtype = torch.float))
		print("IWAE Loss for z", -iwae_loss)
	do_plot = False
	if do_plot:
		if data=='mnist':
			plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "z-output.png")
		else:
			imputation = b_data.to(device,dtype = torch.float) 
			imputation[~b_mask] = all_logits_obs_model[~b_mask]
			img = imputation.cpu().data.numpy()
			plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "z-output.png")

		prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 
		plot_images_in_row(num_epochs, loc1 =  prefix +str(0) + "z-output.png", loc2 =  prefix +str(a) + "z-output.png", loc3 =  prefix +str(2*a) + "z-output.png", loc4 =  prefix +str(3*a) + "z-output.png", loc5 =  prefix +str(4*a) + "z-output.png", file =  prefix + "z-output-all.png", data=data) 

		if data=='mnist':
			display_images(decoder, q_zgivenxobs, d, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '.png', k = 50)
		else:
			display_images_svhn(decoder, q_zgivenxobs, d, results  + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '.png', k = 50)

	if return_imputation:
		return z_loss, z_mse, -iwae_loss, z_params, img
	else:
		return z_loss, z_mse, -iwae_loss, z_params


def optimize_z_with_labels(num_epochs, p_z, p_y, b_data, b_full, b_mask , encoder, decoder, discriminative_model, device, d, results, iterations, nb, K_samples, data='mnist'):

	r1 = -1
	r2 = 1
	logits_y , means, scales  = init_params_labels(encoder, decoder, discriminative_model, p_z, b_data, b_mask, d, r1, r2, data=data)
	i=-1

	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	means = means.to(device)
	scales = scales.to(device)
	logits_y = logits_y.to(device)

	means.requires_grad = True
	scales.requires_grad = True
	logits_y.requires_grad = True

	test_optimizer_z = torch.optim.Adam([means,scales], lr=1.0, betas=(0.9, 0.999)) 
	test_optimizer_logits = torch.optim.Adam([logits_y], lr=0.01, betas=(0.9, 0.999))  #1.0 for re-inits

	#lambda1 = lambda epoch: 0.67 ** epoch
	lambda1 = lambda epoch: 0.95
	#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers, lr_lambda=lambda1)
	scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer_z, step_size=30, gamma=0.5)
	scheduler_logits = torch.optim.lr_scheduler.StepLR(test_optimizer_logits, step_size=50, gamma=0.5)

	#scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer_z, step_size=30, gamma=0.1)

	z_elbo = np.zeros((num_epochs))
	z_mse = np.zeros((num_epochs))
	z_loglikelihood = np.zeros((num_epochs))
	z_loss= np.zeros((num_epochs))

	a = int(num_epochs/4)
	y_kl = np.zeros((num_epochs)) 
	z_loglike_loss = np.zeros((num_epochs)) 
	z_kl = np.zeros((num_epochs)) 

	for k in range(num_epochs):
		test_optimizer_z.zero_grad()
		test_optimizer_logits.zero_grad()
		loss, log_like, aa, bb, cc = z_loss_with_labels(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, p_y = p_y, means = means, scales = scales, logits_y = logits_y, encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data)

		y_kl[ k] = bb.item()
		z_loglike_loss[ k] = aa.item()
		z_kl[ k] = cc.item()

		loss.backward() 
		test_optimizer_z.step()
		test_optimizer_logits.step()
		scheduler.step()
		#if k>=10:
		#scheduler_logits.step()
		#loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, xm_params = xm_params.to(device,dtype = torch.float), encoder = encoder, decoder = decoder , device = device, d=d, K=100, K_z=1)

		z_loss[k] = loss.item()
		
		#print(z_loss[k])
		##Imputations
		impute = False

		if impute:
			if (k)%a==0 or k ==0:
				if data=='mnist':
					plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "z-output.png")
				else:
					imputation[~b_mask] = all_logits_obs_model[~b_mask]
					#print(all_logits_obs_model[~b_mask])
					img = imputation.cpu().data.numpy()
					plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "z-output.png")

	means.requires_grad = False
	scales.requires_grad = False
	logits_y.requires_grad = False

	print("k , loss", k , loss.item())
	plot_curve(y_kl, num_epochs, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '-kl_y-labels.png' )
	plot_curve(z_loglike_loss, num_epochs, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '-loglike-labels.png' )
	plot_curve(z_kl, num_epochs, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '-kl_z-labels.png' )


	iwae_loss, log_like, aa, bb, cc = z_loss_with_labels(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, p_y = p_y, means = means, scales = scales, logits_y = logits_y, encoder = encoder, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, evaluate=True, full = b_full)

	print("IWAE_bound for M2_z", -iwae_loss)

	print("q_y after ---", torch.nn.functional.softmax(logits_y, dim=1))

	if impute:
		if data=='mnist':
			plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "z-output.png")
		else:
			imputation = b_data.to(device,dtype = torch.float) 
			imputation[~b_mask] = all_logits_obs_model[~b_mask]
			img = imputation.cpu().data.numpy()
			plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "z-output.png")

		prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 
		plot_images_in_row(num_epochs, loc1 =  prefix +str(0) + "z-output.png", loc2 =  prefix +str(a) + "z-output.png", loc3 =  prefix +str(2*a) + "z-output.png", loc4 =  prefix +str(3*a) + "z-output.png", loc5 =  prefix +str(4*a) + "z-output.png", file =  prefix + "z-output-all.png", data=data) 

	if data=='mnist':
		display_images_with_labels(decoder, means, torch.nn.Softplus()(scales), logits_y, d, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'labels.png', k = 50,b_data=b_data, b_mask = b_mask, directory2 = results + str(i) + "/compiled/" + str(nb%10)  + str(iterations) + 'component', file2='labels.png' )
	else:
		display_images_svhn(decoder, q_zgivenxobs, d, results  + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + '.png', k = 50)

	return z_loss, z_mse


def get_pxgivenz(out_encoder, d, data, decoder):
	q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

	zgivenx = q_zgivenxobs.rsample() #K with z_params and without K if out_encoder is used
	zgivenx_flat = zgivenx.reshape([50,d])
	all_logits_obs_model = decoder.forward(zgivenx_flat)

	data_flat = data.reshape([-1,1])
	all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
	all_log_pxgivenz = all_log_pxgivenz_flat.reshape([50,28*28])
	logpxobsgivenz = torch.sum(all_log_pxgivenz,1).reshape([50]) 

	return torch.mean(logpxobsgivenz)

def optimize_mixture(num_epochs, p_z, b_data, b_full, b_mask , encoder, decoder, device, d, results, iterations, nb, K_samples, data='mnist', with_labels=False, labels = None, do_random=True, return_imputation=False):

	i=-1
	batch_size = b_data.shape[0]
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	num_components = 10

	b_data[~b_mask] = 0
	r1 = -1
	r2 = 1
	#print(b_data.shape, type(b_data)) 

	logits, means, scales = init_mixture(encoder, decoder, p_z, b_data, b_mask, num_components, batch_size, d, r1, r2, data=data)
	do_plot = False
	logits = logits.to(device,dtype = torch.float)
	means = means.to(device,dtype = torch.float)
	scales = scales.to(device,dtype = torch.float)

	if do_plot:
		comp_samples = np.zeros((num_components+2,50,d))
		for comp in range(num_components):
			probs = torch.zeros(batch_size, num_components)
			probs[0,comp] = 1.0
			dist = td.MixtureSameFamily(td.Categorical(probs=probs.to(device,dtype = torch.float)), td.Independent(td.Normal(means, scales.exp()), 1))
			#print(dist.sample([50]).cpu().data.numpy(), dist.sample([50]).cpu().data.numpy().shape)
			a = dist.sample([50])
			comp_samples[comp] = a.cpu().data.numpy().reshape([50,d]) 
			display_images(decoder, dist, d, results + str(i) + "/compiled/components/" + str(nb%10)  + str(iterations)  + '-component' + str(comp) + 'init-mixture.png', k = 50, data=data)

		if data == 'mnist':
			x_7 = get_sample_digit(data_dir = "data" , digit=7, file = results + str(i) + "/compiled/")
			x_9 = get_sample_digit(data_dir = "data" , digit=9 ,file = results + str(i) + "/compiled/")
			#x_7_embed = = encoder.forward(x_7.to(device,dtype = torch.float))
			#x_
			out_7 = encoder.forward(x_7.to(device,dtype = torch.float))
			out_9 = encoder.forward(x_9.to(device,dtype = torch.float))
			comp_samples[num_components] =  out_7[...,:d].cpu().data.numpy().reshape([50,d]) 
			comp_samples[num_components+1] = out_9[...,:d].cpu().data.numpy().reshape([50,d]) 
			print(get_pxgivenz(out_7,d, x_7.to(device,dtype = torch.float),decoder), get_pxgivenz(out_9,d, x_9.to(device,dtype = torch.float), decoder))

		comp_samples = comp_samples.reshape(((num_components+2)*50,d))

		X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(comp_samples)
		scatter_plot_(X.reshape((num_components+2,50,2)), num_components, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'init-mixture-scatterplot.png')

		print("entropy before------", get_entropy_mixture(logits))
		print(td.Independent(td.Normal(means, scales.exp()), 1).entropy())
		
	#print(td.MixtureSameFamily(td.Categorical(logits=logits), td.Independent(td.Normal(means.detach(), scales.exp().detach()), 1)).entropy())
		
	logits.requires_grad = True
	means.requires_grad = True
	scales.requires_grad = True
	
	#test_optimizer = torch.optim.LBFGS([xm_params]) ##change, sgd, adagrad
	test_optimizer = torch.optim.Adam([means,scales], lr=1.0, betas=(0.9, 0.999)) 
	test_optimizer_logits = torch.optim.Adam([logits], lr=0.1, betas=(0.9, 0.999))  #1.0 for re-inits

	#lambda1 = lambda epoch: 0.67 ** epoch
	lambda1 = lambda epoch: 0.95
	#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers, lr_lambda=lambda1)
	scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=30, gamma=0.5)
	scheduler_logits = torch.optim.lr_scheduler.StepLR(test_optimizer_logits, step_size=30, gamma=0.5)
	#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(test_optimizer, lr_lambda=lambda1)
	#scheduler_logits = torch.optim.lr_scheduler.MultiplicativeLR(test_optimizer_logits, lr_lambda=lambda1)

	lrs = []

	#test_optimizer = torch.optim.Adagrad([xm_params], lr=0.1) 
	#b_data = b_data_init
	#q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d],scale=torch.nn.Softplus()(z_params[...,d:])),1)

	z_elbo = np.zeros((num_epochs))
	z_mse = np.zeros((num_epochs))
	z_loglikelihood = np.zeros((num_epochs))
	z_loss= np.zeros((num_epochs))

	a = int(num_epochs/4)
	beta = 1

	#do_random = True
	do_plot = False
	#p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(logits=torch.zeros(batch_size, num_components).cuda()), td.Independent(td.Normal(torch.zeros(batch_size, num_components, d).cuda() + torch.rand(batch_size, num_components, d).cuda(), torch.ones(batch_size, num_components, d).cuda()), 1))

	#p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(logits=torch.zeros(batch_size, num_components).cuda()), td.Independent(td.Normal(torch.zeros(batch_size, num_components, d).cuda(), torch.ones(batch_size, num_components, d).cuda()), 1))

	for k in range(num_epochs):
		##Let the optimization find modes first
		if not do_random and k==0:
			logits.requires_grad = False

		if do_random and k==99:
			logits.requires_grad = False

		##Adjust the weights of the modes in the mixture
		if k==199:
			logits.requires_grad = True

		##Reset learning rate for means and scales
		if do_random and k==99:
			del test_optimizer, scheduler
			test_optimizer = torch.optim.Adam([means,scales], lr=1.0, betas=(0.9, 0.999)) 
			scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=30, gamma=0.5)

		test_optimizer.zero_grad()
		test_optimizer_logits.zero_grad()
		loss, log_like, aa, bb = mixture_loss(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, means = means, scales = scales, logits = logits,  decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta)
		loss.backward() 
		test_optimizer.step()
		test_optimizer_logits.step()

		lrs.append(test_optimizer.param_groups[0]["lr"])
		scheduler.step()

		if k>=199:
			scheduler_logits.step()
		#loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, xm_params = xm_params.to(device,dtype = torch.float), encoder = encoder, decoder = decoder , device = device, d=d, K=100, K_z=1)

		z_loss[k] = loss.item()

		#Impute 
		if do_plot:
			with torch.no_grad():
				imputation = b_data.to(device,dtype = torch.float) 
				q_z = ReparameterizedNormalMixture1d(logits.detach(), means.detach(),torch.nn.Softplus()(scales))
				zgivenx = q_z.sample([])
				zgivenx_flat = zgivenx.reshape([1,d])
				all_logits_obs_model = decoder.forward(zgivenx_flat)
				if data=='mnist':
					imputation[~b_mask] = torch.sigmoid(all_logits_obs_model)[~b_mask]
				else:
					imputation[~b_mask] = all_logits_obs_model[~b_mask]

				#G/et error on imputation
				img = imputation.cpu().data.numpy()
				err = np.array([mse(img.reshape([1,channels,p,q]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
				z_mse[k] = err

				if (k)%a==0 or k ==0:
					if data=='mnist':
						plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "mixture.png")
					else:
						imputation[~b_mask] = all_logits_obs_model[~b_mask]
						#print(all_logits_obs_model[~b_mask])
						img = imputation.cpu().data.numpy()
						plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "mixture.png")

		ap = False
		if do_random and k%20==0 and k<=100: #
			#print("k ---", k )
			threshold = 0.01
			probs = torch.softmax(logits.detach(),dim=1)
			#print(probs)
			ap = torch.any(probs<threshold)
			#print(ap)
			if ap:
				b_data[~b_mask] = 0
				logits_, means_, scales_ = init_mixture(encoder, decoder, p_z, b_data, b_mask, num_components, batch_size, d, r1, r2, data=data, repeat=True)
				logits_ = logits_.to(device,dtype = torch.float)
				means_ = means_.to(device,dtype = torch.float)
				scales_ = scales_.to(device,dtype = torch.float)
				#logits = torch.where(probs > threshold, logits.detach(), logits_)

		#For the first 100 iterations, re-init means 
		if do_random and k%20==0 and k<=100 and ap:
			means__ = means.detach()
			scales__ = scales.detach()
			logits = logits_
			probs = torch.Tensor.repeat(probs.reshape(batch_size, num_components, 1),[1,1,d]) 
			means = torch.where(probs >= threshold, means__, means_)
			scales = torch.where(probs >= threshold, scales__, scales_)
			#print(logits, logits_)
			logits.requires_grad = True
			means.requires_grad = True
			scales.requires_grad = True

			lr_ = test_optimizer.param_groups[0]["lr"]
			del test_optimizer
			test_optimizer = torch.optim.Adam([means,scales], lr=lr_, betas=(0.9, 0.999)) ##Ask Gabe about any alternative

			lr_ = test_optimizer_logits.param_groups[0]["lr"]
			del test_optimizer_logits 
			test_optimizer_logits = torch.optim.Adam([logits], lr=0.1, betas=(0.9, 0.999)) ##Ask Gabe about any alternative

	logits.requires_grad = False
	means.requires_grad = False
	scales.requires_grad = False
	#plt.plot(range(num_epochs),lrs)
	#plt.show()
	#plt.savefig(results + str(i) + "/compiled/components/" + str(nb%10)  + str(iterations)  + 'lr.png')
	#plt.close()

	iwae_loss = 0
	iwae_loss, log_like, aa, bb = mixture_loss(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, means = means, scales = scales, logits = logits,  decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta, evaluate=True, full= b_full.to(device,dtype = torch.float))

	if do_plot:
		iwae_loss, log_like, aa, bb = mixture_loss(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, means = means, scales = scales, logits = logits,  decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta, evaluate=True, full= b_full.to(device,dtype = torch.float))

	print("IWAE Loss for mixture", -iwae_loss)
	#print("means after --", torch.sum(means,2))
	
	comp_samples = 0
	
	do_plot=False
	if do_plot:
		comp_samples = np.zeros((num_components+2,50,d))
		for comp in range(num_components):
			probs = torch.zeros(batch_size, num_components)
			probs[0,comp] = 1.0
			dist = td.MixtureSameFamily(td.Categorical(probs=probs.to(device,dtype = torch.float)), td.Independent(td.Normal(means.detach(), scales.exp().detach()), 1))
			#print(dist.sample([50]).cpu().data.numpy(), dist.sample([50]).cpu().data.numpy().shape)
			comp_samples[comp] = dist.sample([50]).cpu().data.numpy().reshape([50,d])
			display_images(decoder, dist, d, results + str(i) + "/compiled/components/" + str(do_random) + str(nb%10)  + str(iterations)  + '-component' + str(comp) + 'mixture.png', k = 50, data=data)
		
		#if data== 'mnist':
		#	comp_samples[num_components] =  out_7[...,:d].cpu().data.numpy().reshape([50,d]) 
		#	comp_samples[num_components+1] = out_9[...,:d].cpu().data.numpy().reshape([50,d]) 

		#comp_samples = comp_samples.reshape([(num_components+2)*50,d])
		#X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(comp_samples)
		#scatter_plot_(X.reshape([num_components+2,50,2]), num_components, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'mixture-scatterplot.png')

		print("entropy after------", get_entropy_mixture(logits))
		print(td.Independent(td.Normal(means, scales.exp()), 1).entropy())
		#print(td.MixtureSameFamily(td.Categorical(logits=logits), td.Independent(td.Normal(means.detach(), scales.exp().detach()), 1)).entropy())
		#print(scales.exp())
		#if data=='mnist':
		#	plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "mixture.png")
		#else:
		#	imputation = b_data.to(device,dtype = torch.float) 
		#	imputation[~b_mask] = all_logits_obs_model[~b_mask]
		#	img = imputation.cpu().data.numpy()
		#	plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(num_epochs) + "mixture.png")

		#prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 
		#plot_images_in_row(num_epochs, loc1 =  prefix +str(0) + "mixture.png", loc2 =  prefix +str(a) + "mixture.png", loc3 =  prefix +str(2*a) + "mixture.png", loc4 =  prefix +str(3*a) + "mixture.png", loc5 =  prefix +str(4*a) + "mixture.png", file =  prefix + "mixture-all.png", data=data) 

		q_z = ReparameterizedNormalMixture1d(logits, means, torch.nn.Softplus()(scales))

		if data=='mnist':
			display_images(decoder, q_z, d, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + str(do_random)+'mixture.png', k = 50, b_data = b_data, b_mask=b_mask)
		else:
			display_images_svhn(decoder, q_z, d, results  + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + str(do_random) +'mixture.png', k = 50)

	del test_optimizer, test_optimizer_logits, scheduler, scheduler_logits, comp_samples

	if return_imputation: 
		return z_loss, z_mse, -iwae_loss, logits, means, scales
	else:
		return z_loss, z_mse, -iwae_loss, logits, means, scales


def optimize_mixture_labels(num_epochs, p_z,  p_y, b_data, b_full, b_mask, encoder, decoder, discriminative_model, device, d, results, iterations, nb, K_samples, data='mnist', labels = None, do_random=False):
	i=-1
	batch_size = b_data.shape[0]
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	num_components = 10

	b_data[~b_mask] = 0
	r1 = -0.5
	r2 = 0.5
	#print(b_data.shape, type(b_data)) 

	logits, means, scales, logits_y = init_params_mixture_labels(encoder, decoder, discriminative_model,  b_data, b_mask, num_components, batch_size, d, r1, r2, data=data)

	logits = logits.to(device,dtype = torch.float)
	means = means.to(device,dtype = torch.float)
	scales = scales.to(device,dtype = torch.float)
	logits_y = logits_y.to(device,dtype = torch.float)

	#print("initial class probs ---", torch.nn.functional.softmax(logits_y, dim=1))

	logits.requires_grad = False
	means.requires_grad = True
	scales.requires_grad = True
	logits_y.requires_grad = False #False if random=True

	#test_optimizer = torch.optim.LBFGS([xm_params]) ##change, sgd, adagrad
	test_optimizer = torch.optim.Adam([means,scales], lr=1.0, betas=(0.9, 0.999)) 
	test_optimizer_logits = torch.optim.Adam([logits], lr=0.1, betas=(0.9, 0.999))  #1.0 for re-inits
	test_optimizer_logits_y = torch.optim.Adam([logits_y], lr=0.01, betas=(0.9, 0.999)) 

	#lambda1 = lambda epoch: 0.67 ** epoch
	lambda1 = lambda epoch: 0.95
	#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers, lr_lambda=lambda1)
	scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=100, gamma=0.5)
	scheduler_logits = torch.optim.lr_scheduler.StepLR(test_optimizer_logits, step_size=30, gamma=0.5)
	#scheduler_logits_y = torch.optim.lr_scheduler.StepLR(test_optimizer_logits_y, step_size=30, gamma=0.5)

	#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(test_optimizer, lr_lambda=lambda1)
	#scheduler_logits = torch.optim.lr_scheduler.MultiplicativeLR(test_optimizer_logits, lr_lambda=lambda1)

	lrs = []

	z_elbo = np.zeros((num_epochs))
	z_mse = np.zeros((num_epochs))
	z_loglikelihood = np.zeros((num_epochs))
	z_loss= np.zeros((num_epochs))

	a = int(num_epochs/4)
	beta = 1

	for k in range(num_epochs):

		if do_random and k==0:
			logits.requires_grad = True
			logits_y.requires_grad = True

		##Let the optimization find components in a label first
		if do_random and k==99:
			logits.requires_grad = False
			#logits_y.requires_grad = False

		##Adjust the weights of the modes in the mixture
		if do_random:
			if k==199:
				#logits.requires_grad = True
				logits_y.requires_grad = True
		else:
			if k==99:
				logits.requires_grad = True
				logits_y.requires_grad = True			

		##Reset learning rate for means and scales
		if do_random and k==99:
			del test_optimizer, scheduler
			test_optimizer = torch.optim.Adam([means,scales], lr=1.0, betas=(0.9, 0.999)) 
			scheduler = torch.optim.lr_scheduler.StepLR(test_optimizer, step_size=50, gamma=0.5)

		if do_random and k==199:
			del test_optimizer_logits
			test_optimizer_logits = torch.optim.Adam([logits], lr=0.1, betas=(0.9, 0.999)) 
			scheduler_logits = torch.optim.lr_scheduler.StepLR(test_optimizer_logits, step_size=30, gamma=0.5)

		test_optimizer.zero_grad()
		test_optimizer_logits.zero_grad()
		test_optimizer_logits_y.zero_grad()
		loss, log_like, aa, bb = mixture_loss_labels(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z,  means = means, scales = scales, logits = logits, logits_y=logits_y, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta, num_components=num_components, iwae=False)
		loss.backward() 
		test_optimizer.step()
		test_optimizer_logits.step()
		test_optimizer_logits_y.step()
		lrs.append(test_optimizer.param_groups[0]["lr"])
		scheduler.step()

		if k>=199:
			scheduler_logits.step()

		#scheduler_logits_y.step()
		#loss, log_like, aa, bb, z_params = z_loss_(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, z_params = z_params, xm_params = xm_params.to(device,dtype = torch.float), encoder = encoder, decoder = decoder , device = device, d=d, K=100, K_z=1)

		z_loss[k] = loss.item()

		#Impute 
		impute = False
		if impute:
			imputation = b_data[:,0,:,:].reshape([1,1,p,q]).to(device,dtype = torch.float) 
			q_z = ReparameterizedNormalMixture1d(logits.detach(), means.detach(), scales.exp().detach())
			q_y = td.Categorical(logits=logits_y)

			zgivenx = q_z.sample([])
			ygivenx = q_y.sample([])

			b =  torch.zeros(1,10)
			b[0, ygivenx] = 1
			zgivenx_y = torch.cat((b.to(device,dtype = torch.float)),2)
			zgivenx_y = torch.cat((zgiveny,label.to(device,dtype = torch.float)),1)
			zgivenx_flat = zgivenx_y.reshape([1,d+10])
			all_logits_obs_model = decoder.forward(zgivenx_flat)

			if data=='mnist':
				imputation[~b_mask] = torch.sigmoid(all_logits_obs_model)[~b_mask]
			else:
				imputation[~b_mask] = all_logits_obs_model[~b_mask]

			#G/et error on imputation
			img = imputation.cpu().data.numpy()
			err = np.array([mse(img.reshape([1,channels,p,q]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
			z_mse[k] = err

			if (k)%a==0 or k ==0:
				if data=='mnist':
					plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "mixture.png")
				else:
					imputation[~b_mask] = all_logits_obs_model[~b_mask]
					#print(all_logits_obs_model[~b_mask])
					img = imputation.cpu().data.numpy()
					plot_image_svhn(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "mixture.png")

		ap = False
		if do_random and k%20==0 and k<=100: #
			#print("k ---", k )
			threshold = 0.015
			probs = torch.softmax(logits.detach(),dim=1)
			#print(probs)
			ap = torch.any(probs<threshold)
			#print(ap)
			if ap:
				b_data[~b_mask] = 0
				logits_, means_, scales_, logits_y_ = init_params_mixture_labels(encoder, decoder, discriminative_model, b_data, b_mask, num_components, batch_size, d, r1, r2, data=data)
				logits_ = logits_.to(device,dtype = torch.float)
				means_ = means_.to(device,dtype = torch.float)
				scales_ = scales_.to(device,dtype = torch.float)
				logits_y_ = logits_y_.to(device,dtype = torch.float)
				#logits = torch.where(probs > threshold, logits.detach(), logits_)

		#For the first 100 iterations, re-init means 
		if do_random and k%20==0 and k<=100 and ap:
			means__ = means.detach()
			scales__ = scales.detach()
			logits = logits_
			logits_y = logits_y_

			probs = torch.Tensor.repeat(probs.reshape(10, num_components, 1),[1,1,d]) 
			means = torch.where(probs >= threshold, means__, means_)
			scales = torch.where(probs >= threshold, scales__, scales_)
			#print(logits, logits_)
			logits.requires_grad = True
			means.requires_grad = True
			scales.requires_grad = True
			logits_y.requires_grad = False

			lr_ = test_optimizer.param_groups[0]["lr"]
			del test_optimizer
			test_optimizer = torch.optim.Adam([means,scales], lr=lr_, betas=(0.9, 0.999))

			lr_ = test_optimizer_logits.param_groups[0]["lr"]
			del test_optimizer_logits 
			test_optimizer_logits = torch.optim.Adam([logits], lr=0.1, betas=(0.9, 0.999)) 

			lr_ = test_optimizer_logits_y.param_groups[0]["lr"]
			del test_optimizer_logits_y 
			test_optimizer_logits_y = torch.optim.Adam([logits_y], lr=0.01, betas=(0.9, 0.999)) 

		#print(logits_y)

	logits.requires_grad = False
	means.requires_grad = False
	scales.requires_grad = False
	logits_y.requires_grad = False #False if random=True

	print("k , loss", k , loss.item())
	iwae_loss, log_like, aa, bb = mixture_loss_labels(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z,  means = means, scales = scales, logits = logits, logits_y=logits_y, decoder = decoder , device = device, d=d, K=K_samples, K_z=1, data=data, beta=beta, num_components=num_components, iwae=False, evaluate=True, full = b_full)
	

	plt.plot(range(num_epochs),lrs)
	plt.show()
	plt.savefig(results + str(i) + "/compiled/components/" + str(nb%10)  + str(iterations)  + 'lr.png')
	plt.close()

	#print("means after --", torch.sum(means,2))
	#print("Mixture weights after --", torch.nn.functional.softmax(logits, dim=1) )
	#print("q_y after ---", torch.nn.functional.softmax(logits_y, dim=1))
	print("IWAE for M2 + mixture : ", -iwae_loss)
	if data=='mnist':
		display_images_with_labels(decoder, means, torch.nn.Softplus()(scales), logits_y, d, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'mixture_labels.png', k = 50,b_data=b_data, b_mask = b_mask, directory2 = results + str(i) + "/compiled/" + str(nb%10)  + str(iterations) + 'component', file2 = str(do_random)+'mixture_labels.png', logits= logits)
	else:
		display_images_svhn(decoder, q_z, d, results  + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'mixture.png', k = 50)


	del logits,means,scales, test_optimizer, test_optimizer_logits, scheduler, scheduler_logits #, comp_samples
	return z_loss, z_mse


