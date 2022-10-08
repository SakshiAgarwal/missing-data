import torch
import numpy as np
from loss import *
import gc
import torch.distributions as td
import os
from plot import *


def burn_in(b_data, b_mask, labels, encoder, decoder, p_z, d, burn_in_period=20, data='mnist', with_labels=False):
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]
	if data=='mnist':
		x_logits = torch.log(b_data/(1-b_data)).reshape(1,channels,p,q)
	else:
		x_logits = b_data.reshape(1,channels,p,q)

	if data=='mnist':
		z_init =  mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1, with_labels=with_labels, labels= labels)[1]
		for l in range(burn_in_period):
			x_logits, z_init = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1, with_labels=with_labels, labels= labels)
			x_logits = x_logits.reshape(1,1,p,q)
			b_data[0,0,:,:].reshape([1,1,28,28])[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
	else: 
		z_init =  mvae_impute_svhn(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1)[1]
		for l in range(burn_in_period):
			x_logits, z_init, sigma_decoder = mvae_impute_svhn(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1)
			x_logits = x_logits.reshape(1,channels,p,q)
			b_data[~b_mask] = td.Normal(loc = x_logits, scale =  sigma_decoder.exp()*(torch.ones(*x_logits.shape).cuda())).sample()[~b_mask]

	return x_logits, b_data, z_init

def burn_in_table(b_data, b_mask, encoder, decoder, p_z, d, burn_in_period=20):
	p = b_data.shape[1]
	x_logits = b_data.reshape(1,p)

	for l in range(burn_in_period):
		x_logits, sample = miwae_impute_uci(b_data, b_mask, encoder ,decoder,p_z, d, L=1, return_sample=True)
		b_data[0,:].reshape([1,p])[~b_mask] = sample[~b_mask]

	return x_logits, b_data

def burn_in_svhn_trash(b_data, b_mask, model, p_z, d, burn_in_period=20, data='mnist', with_labels=False):
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]

	x_logits = b_data.reshape(1,channels,p,q)

	x_logits, mu, std =  mvae_impute_svhn(iota_x = b_data,mask = b_mask,model = model, p_z = p_z, d=d, L=1)

	for l in range(burn_in_period):
		x_logits, mu, std = mvae_impute_svhn(iota_x = b_data,mask = b_mask, model = model, p_z = p_z, d=d, L=1)
		x_logits = x_logits.reshape(1,channels,p,q)
		b_data[~b_mask] = td.Normal(loc = x_logits, scale =  (torch.ones(*x_logits.shape).cuda())).sample()[~b_mask]

	return x_logits, b_data,  mu, std

def burn_in_svhn(b_data, b_mask, encoder, decoder, p_z, d, burn_in_period=20, data='mnist', with_labels=False):
	channels = b_data.shape[1]
	p = b_data.shape[2]
	q = b_data.shape[3]
	results=os.getcwd() + "/results/svhn/" 
	#print("in burn-in")
	x_logits = b_data.reshape(1,channels,p,q)
    
	#z_init =  mvae_impute_svhn(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1)[1]
	for l in range(burn_in_period):
		img = b_data.cpu().data.numpy()    
		#print("1st iteration:", torch.mean(b_data[:,0,:,:][~b_mask[:,0,:,:]]), torch.mean(b_data[:,1,:,:][~b_mask[:,1,:,:]]), torch.mean(b_data[:,2,:,:][~b_mask[:,2,:,:]]))
		plot_image_svhn(np.squeeze(img),results + str(-1) + "/compiled/" + str(l) + "-burn-in.png" )
		x_logits, z_init, sigma_decoder, sample = mvae_impute_svhn(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1, return_sample=True)
		#x_logits = x_logits.reshape(1,channels,p,q)
		b_data[~b_mask] = sample[~b_mask]
		imputation = b_data
		imputation[~b_mask] = x_logits[~b_mask]
        
	return x_logits, b_data, z_init


def burn_in_noise(b_data, b_mask, encoder, decoder, p_z,  d, burn_in_period=20, sigma=0.1, nb=0):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_size = b_data.shape[0]
	#x_logits_prev = torch.zeros_like(b_data).to(device,dtype = torch.double)
	mp = len(b_data[~b_mask])
	#print("len og gaussian pixels: ", mp)

	start = 0.5 + 0*torch.clamp(b_data, min=0, max=1).to(device, dtype = torch.float)
	x_logits_prev = torch.log(start/(1-start))
	#p_x_m = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_prev[~b_mask]),1)
	#x_prev = p_x_m.sample().reshape(1,batch_size,28,28)

	#px = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_prev[~b_mask]),1)
	#x_prev = px.sample().reshape(1,batch_size,28,28).to(device, dtype = torch.double)
	x_prev = start
	y_prev = b_data

	K=1

	results=os.getcwd() + "/results/mnist-False-" 

	for l in range(burn_in_period):
		#print(l)
		out_encoder = encoder.forward(x_prev)
		q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
		zgivenx = q_zgivenxobs.rsample()
		zgivenx_flat = zgivenx.reshape([batch_size,d])
		x_logits = decoder.forward(zgivenx_flat)
		xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits.reshape([-1,1])),1)

		for c in range(1):
			x_t = xgivenz.sample().reshape(1,batch_size,28,28).cuda()
			plot_image(np.squeeze(x_t.cpu().data.numpy()),results + str(-2) + "/images/" + str(nb%10) + "/"  + str(l) + "-proposedsample.png" )

			logpy_x_t = torch.zeros(mp).to(device,dtype = torch.float)
			logpy_x_prev = torch.zeros(mp).to(device,dtype = torch.float)
			log_rho_x = torch.zeros(mp).to(device,dtype = torch.float)

			for pixel in range(mp):
				#print(pixel)
				pygivenx = td.Normal(loc=x_t[~b_mask][pixel],scale=sigma)
				pygivenx_prev = td.Normal(loc=x_prev[~b_mask][pixel],scale=sigma)
				#pygivenx = torch.distributions.multivariate_normal.MultivariateNormal(x_t.reshape([-1]).cuda(), covariance_matrix=torch.mul(torch.eye(28*28), pow(sigma,2)).cuda())#covariance_matrix=torch.eye(28*28).cuda()) ## rescale sigma
				#pygivenx_prev = torch.distributions.multivariate_normal.MultivariateNormal(x_prev.reshape([-1]).cuda(), covariance_matrix=torch.mul(torch.eye(28*28), pow(sigma,2)).cuda())

				#print(torch.mul(torch.eye(28*28), pow(sigma,2)), torch.mul(torch.eye(28*28), pow(sigma,2)).size(), x_t.reshape([-1,1]).size())
				#print("calculating --", pygivenx.log_prob(y_prev.reshape([-1])[pixel].cuda()))
				logpy_x_t[pixel] = pygivenx.log_prob(y_prev[~b_mask][pixel].cuda()) #
				logpy_x_prev[pixel] = pygivenx_prev.log_prob(y_prev[~b_mask][pixel].cuda()) #.reshape(-1)
				log_rho_x[pixel] =  logpy_x_t[pixel] - logpy_x_prev[pixel]
				#print(x_t.reshape(-1)[pixel], x_prev.reshape(-1)[pixel])
				#print(logpy_x_t[pixel], logpy_x_prev[pixel], log_rho_x[pixel] )
			
			#log_rho_x = log_rho_x.to(device,dtype = torch.float)
			#print(log_rho_x)
			#print(sum(log_rho_x > torch.zeros(mp).to(device,dtype = torch.float)))

			
			log_rho_x = torch.where(log_rho_x > torch.zeros(mp).to(device,dtype = torch.float), torch.tensor(0).to(device,dtype = torch.float) , log_rho_x)
			aa = torch.rand(mp).to(device,dtype = torch.float)
			x_prev[~b_mask] = torch.where(aa < torch.exp(log_rho_x), x_t[~b_mask], x_prev[~b_mask]) ##.reshape([1,1,28,28])
			x_logits_prev[~b_mask] = torch.where(aa < torch.exp(log_rho_x), x_logits[~b_mask] , x_logits_prev[~b_mask]) ##.reshape(-1)).reshape([1,1,28,28])

			x_prev[b_mask] = x_t[b_mask]
			x_logits_prev[b_mask] = x_logits[b_mask]
			#x_prev[ ] = x_t
			#x_logits[aa > torch.exp(log_rho_x)] = x_logits_prev
			#print(sum(aa < torch.exp(log_rho_x)))
			#if aa < torch.exp(log_rho_x):
			#	x_prev = x_t
			#	x_logits = x_logits
			#	print("accepted")
			#else:
			#	x_prev = x_prev
			#	x_logits = x_logits_prev
			#	print("rejected")

			data_flat = x_prev.reshape([-1,1]).cuda()
			all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_prev.reshape([-1,1])).log_prob(data_flat)
			all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,28*28])
			logpxobsgivenz = torch.sum(all_log_pxgivenz,1).reshape([K,batch_size])

			a = torch.sigmoid(x_logits_prev)
			plot_image(np.squeeze(a.cpu().data.numpy()),results + str(-2) + "/images/" + str(nb%10) + "/"  + str(l) + "-init.png" )

	#exit()


	##check how the burn-in image changes

	#b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
	#b_data = x_prev
	return x_logits_prev, x_prev

def k_neighbors_sample(iota_x, mask):
	p = iota_x.shape[2]
	iota_x = iota_x.cpu().data
	mask = mask.cpu().data

	missing_pixels = torch.nonzero(~torch.squeeze(mask))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	params = torch.zeros_like(iota_x[~mask])
	#params = params.to(device,dtype = torch.float)

	#print("printing --")
	#print(missing_pixels)

	#print(mask[0,0,0,1])

	x = torch.tensor(0)
	#x.to(device,dtype = torch.float)
	for i in range(len(params)):
		pixel = missing_pixels[i]
		k=1

		#print(i,pixel)
		#print(mask[0,0,pixel[0], pixel[1]], iota_x[0,0,pixel[0], pixel[1]])

		while(k<27):
			#print("Trying k =  ----", k)
			aa = ab = ac = ad = True
			left_p = pixel[0] - k
			right_p = pixel[0] + k
			top_p = pixel[1] - k
			bottom_p = pixel[1] + k
			#print("locations --", left_p, right_p, top_p, bottom_p)

			if left_p<0:
				aa = False
				left_p=0
			if right_p>27:
				ab = False
				right_p=27
			if top_p<0:
				ac = False
				top_p=0
			if bottom_p>27:
				ad = False
				bottom_p=27

			a = 0
			num_pixels = 0

			if ac:
				a += sum(mask[0,0,left_p:right_p+1, top_p])
				num_pixels += len(mask[0,0,left_p:right_p+1, top_p])
			if ad:
				a += sum(mask[0,0,left_p:right_p+1, bottom_p])
				num_pixels += len(mask[0,0,left_p:right_p+1, bottom_p])
			if aa:
				a += sum(mask[0,0,left_p, top_p+1:bottom_p])
				num_pixels += len(mask[0,0,left_p, top_p+1:bottom_p]) 
			if ab:
				a += sum(mask[0,0,right_p, top_p+1:bottom_p])
				num_pixels += len(mask[0,0,right_p, top_p+1:bottom_p]) 

			if a>0:
				the_pixel = torch.randint(0,num_pixels,(1,))
				#print("sampled pixel ", the_pixel, " from range ", num_pixels)
				#print(a)

				if ac:
					if (the_pixel >= len(mask[0,0,left_p:right_p+1, top_p])): 
						the_pixel -=  len(iota_x[0,0,left_p:right_p+1, top_p])
					else:			
						x = iota_x[0,0,left_p + the_pixel, top_p]
						params[i] = x
						#print("done ac", params[i])
						break 
				#print(the_pixel)
				if ad:
					if (the_pixel >= len(mask[0,0,left_p:right_p+1, bottom_p])): 
						the_pixel -=  len(iota_x[0,0,left_p:right_p+1, bottom_p])
					else:
						x = iota_x[0,0,left_p + the_pixel, bottom_p]
						params[i] = x
						#print("done ad", params[i])
						break 
				#print(the_pixel)

				if aa:
					if (the_pixel >= len(mask[0,0,left_p, top_p+1:bottom_p])): 
						the_pixel -=  len(iota_x[0,0,left_p, top_p+1:bottom_p])
					else:
						x = iota_x[0,0,left_p, top_p+1+the_pixel]
						params[i] = x
						#print("done aa", params[i])
						break 
				#print(the_pixel)

				if ab:
					if (the_pixel >= len(mask[0,0,right_p, top_p+1:bottom_p])): 
						the_pixel -=  len(iota_x[0,0,right_p, top_p+1:bottom_p])
					else:
						x = iota_x[0,0,right_p, top_p+1+the_pixel]
						params[i] = x
						#print("done ab", params[i])
						break 
			else:
				k += 1
	params = params.to(device,dtype = torch.float)
	return params

def k_neighbors_sample_svhn(iota_x, mask):
	p = iota_x.shape[2]
	iota_x = iota_x.cpu().data
	mask = mask.cpu().data

	#print(torch.squeeze(mask).shape)
	missing_pixels = torch.nonzero(~torch.squeeze(mask))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	params = torch.zeros_like(iota_x[~mask])
	#params = params.to(device,dtype = torch.float)

	#print("printing --")
	#print(missing_pixels)

	#print(mask[0,0,0,1])

	x = torch.tensor(0)
	#x.to(device,dtype = torch.float)
	for i in range(len(params)):
		pixel = missing_pixels[i]
		#print(pixel)
		k=1

		#print(i,pixel)
		#print(mask[0,pixel[0],pixel[1], pixel[2]], iota_x[0,pixel[0],pixel[1], pixel[2]])
		while(k<p-1):
			#print("Trying k =  ----", k)
			aa = ab = ac = ad = True
			left_p = pixel[1] - k
			right_p = pixel[1] + k
			top_p = pixel[2] - k
			bottom_p = pixel[2] + k
			#print("locations --", left_p, right_p, top_p, bottom_p)

			if left_p<0:
				aa = False
				left_p=0
			if right_p>p-1:
				ab = False
				right_p=p-1
			if top_p<0:
				ac = False
				top_p=0
			if bottom_p>p-1:
				ad = False
				bottom_p=p-1

			a = 0
			num_pixels = 0

			if ac:
				a += sum(mask[0,pixel[0],left_p:right_p+1, top_p])
				num_pixels += len(mask[0,pixel[0],left_p:right_p+1, top_p])
			if ad:
				a += sum(mask[0,pixel[0],left_p:right_p+1, bottom_p])
				num_pixels += len(mask[0,pixel[0],left_p:right_p+1, bottom_p])
			if aa:
				a += sum(mask[0,pixel[0],left_p, top_p+1:bottom_p])
				num_pixels += len(mask[0,pixel[0],left_p, top_p+1:bottom_p]) 
			if ab:
				a += sum(mask[0,pixel[0],right_p, top_p+1:bottom_p])
				num_pixels += len(mask[0,pixel[0],right_p, top_p+1:bottom_p]) 

			if a>0:
				the_pixel = torch.randint(0,num_pixels,(1,))
				#print("sampled pixel ", the_pixel, " from range ", num_pixels)
				#print(a)

				if ac:
					if (the_pixel >= len(mask[0,pixel[0],left_p:right_p+1, top_p])): 
						the_pixel -=  len(iota_x[0,pixel[0],left_p:right_p+1, top_p])
					else:			
						x = iota_x[0,pixel[0],left_p + the_pixel, top_p]
						params[i] = x
						#print("done ac", params[i])
						break 
				#print(the_pixel)
				if ad:
					if (the_pixel >= len(mask[0,pixel[0],left_p:right_p+1, bottom_p])): 
						the_pixel -=  len(iota_x[0,pixel[0],left_p:right_p+1, bottom_p])
					else:
						x = iota_x[0,pixel[0],left_p + the_pixel, bottom_p]
						params[i] = x
						#print("done ad", params[i])
						break 
				#print(the_pixel)

				if aa:
					if (the_pixel >= len(mask[0,pixel[0],left_p, top_p+1:bottom_p])): 
						the_pixel -=  len(iota_x[0,pixel[0],left_p, top_p+1:bottom_p])
					else:
						x = iota_x[0,pixel[0],left_p, top_p+1+the_pixel]
						params[i] = x
						#print("done aa", params[i])
						break 
				#print(the_pixel)

				if ab:
					if (the_pixel >= len(mask[0,pixel[0],right_p, top_p+1:bottom_p])): 
						the_pixel -=  len(iota_x[0,pixel[0],right_p, top_p+1:bottom_p])
					else:
						x = iota_x[0,pixel[0],right_p, top_p+1+the_pixel]
						params[i] = x
						#print("done ab", params[i])
						break 
			else:
				k += 1
	params = params.to(device,dtype = torch.float)
	return params


def k_neighbors(iota_x, mask):
	p = iota_x.shape[2]
	iota_x = iota_x.cpu().data
	mask = mask.cpu().data

	#print(torch.squeeze(mask).shape)
	missing_pixels = torch.nonzero(~torch.squeeze(mask))
	params = torch.zeros_like(iota_x[~mask])

	#print("printing --")
	#print(missing_pixels)
	#print(mask[0,0,0,1])

	for i in range(len(missing_pixels)):
		pixel = missing_pixels[i]
		k=1
		#print(i,pixel)
		#print(mask[0,0,pixel[0], pixel[1]], iota_x[0,0,pixel[0], pixel[1]])

		while(k<p-1):
			#print("k---", k )
			aa = ab = ac = ad = True
			left_p = pixel[0] - k
			right_p = pixel[0] + k
			top_p = pixel[1] - k
			bottom_p = pixel[1] + k
			#print("locations --", left_p, right_p, top_p, bottom_p)
			if left_p<0:
				aa = False
				left_p=0
			if right_p>27:
				ab = False
				right_p=27
			if top_p<0:
				ac = False
				top_p=0
			if bottom_p>27:
				ad = False
				bottom_p=27

			a = 0
			num_pixels = 0
			if ac:
				a += sum(mask[0,0,left_p:right_p+1, top_p])
				#num_pixels += len(mask[0,0,left_p:right_p+1, top_p])
				num_pixels += right_p - left_p + 1
			if ad:
				a += sum(mask[0,0,left_p:right_p+1, bottom_p])
				#num_pixels += len(mask[0,0,left_p:right_p+1, bottom_p])
				num_pixels += right_p - left_p + 1
			if aa:
				a += sum(mask[0,0,left_p, top_p+1:bottom_p])
				#num_pixels += len(mask[0,0,left_p, top_p+1:bottom_p]) 
				num_pixels += bottom_p - top_p - 1
			if ab:
				a += sum(mask[0,0,right_p, top_p+1:bottom_p])
				#num_pixels += len(mask[0,0,right_p, top_p+1:bottom_p]) 
				num_pixels += bottom_p - top_p - 1

			if a>0:
				#print(a)
				if ac:
					params[i] += sum(iota_x[0,0,left_p:right_p+1, top_p])
				if ad:
					params[i] += sum(iota_x[0,0,left_p:right_p+1, bottom_p])
				if aa:
					params[i] += sum(iota_x[0,0,left_p, top_p+1:bottom_p])
				if ab:
					params[i] += sum(iota_x[0,0,right_p, top_p+1:bottom_p])

				#print(params[i])
				params[i] = params[i]/a

				if params[i]==0:
					#print("params is 0 here---")
					#params[i]=0.5
					k += 1
				else:
					break 
			else:
				k += 1

	return params

def k_neighbors_svhn(iota_x, mask):
	p = iota_x.shape[2]
	iota_x = iota_x.cpu().data
	mask = mask.cpu().data

	#print(torch.squeeze(mask).shape)
	missing_pixels = torch.nonzero(~torch.squeeze(mask))
	params = torch.zeros_like(iota_x[~mask])

	#print("printing --")
	#print("shape of parameters", params.shape)
	#print(mask[0,0,0,1])

	for i in range(len(missing_pixels)):
		pixel = missing_pixels[i]
		k=1
		#print(i,pixel)
		#print(mask[0,pixel[0],pixel[1], pixel[2]], iota_x[0,pixel[0],pixel[1], pixel[2]])

		while(k<p-1):
			#print("k---", k )
			aa = ab = ac = ad = True
			left_p = pixel[1] - k
			right_p = pixel[1] + k
			top_p = pixel[2] - k
			bottom_p = pixel[2] + k
			#print("locations --", left_p, right_p, top_p, bottom_p)
			if left_p<0:
				aa = False
				left_p=0
			if right_p>p-1:
				ab = False
				right_p=p-1
			if top_p<0:
				ac = False
				top_p=0
			if bottom_p>p-1:
				ad = False
				bottom_p=p-1

			a = 0
			num_pixels = 0
			if ac:
				a += sum(mask[0,pixel[0],left_p:right_p+1, top_p])
				#num_pixels += len(mask[0,0,left_p:right_p+1, top_p])
				num_pixels += right_p - left_p + 1
			if ad:
				a += sum(mask[0,pixel[0],left_p:right_p+1, bottom_p])
				#num_pixels += len(mask[0,0,left_p:right_p+1, bottom_p])
				num_pixels += right_p - left_p + 1
			if aa:
				a += sum(mask[0,pixel[0],left_p, top_p+1:bottom_p])
				#num_pixels += len(mask[0,0,left_p, top_p+1:bottom_p]) 
				num_pixels += bottom_p - top_p - 1
			if ab:
				a += sum(mask[0,pixel[0],right_p, top_p+1:bottom_p])
				#num_pixels += len(mask[0,0,right_p, top_p+1:bottom_p]) 
				num_pixels += bottom_p - top_p - 1

			if a>0:
				#print(a)
				if ac:
					params[i] += sum(iota_x[0,pixel[0],left_p:right_p+1, top_p])
				if ad:
					params[i] += sum(iota_x[0,pixel[0],left_p:right_p+1, bottom_p])
				if aa:
					params[i] += sum(iota_x[0,pixel[0],left_p, top_p+1:bottom_p])
				if ab:
					params[i] += sum(iota_x[0,pixel[0],right_p, top_p+1:bottom_p])

				#print(params[i])
				params[i] = params[i]/a
				#print("parameter updated",params[i])
				#params[i] = 

				break 
			else:
				k += 1

	return params

def init_params_mixture_labels(encoder, decoder, discriminative_model, b_data, b_mask, num_components, batch_size, d, r1, r2, data='mnist'):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logits = torch.zeros(10,10) 
	means = torch.zeros(10, 10, d)
	scales = torch.zeros(10, 10, d)
	#Initialize parameters of q(z|y)

	b_data_ = torch.Tensor.repeat(b_data,[num_components,1,1,1])  
	#print(b_data_.shape)
	b_mask_ = torch.Tensor.repeat(b_mask,[num_components,1,1,1])
	x_logits_init = torch.zeros_like(b_data[:,0,:,:].reshape(1,1,28,28))

	logits_y = torch.zeros(1, 10)

	for i in range(10):
		labels = torch.zeros(num_components, 10)
		labels[:,i] = 1.0
		labels = labels.reshape(num_components, 10, 1, 1)

		labels = torch.repeat_interleave(labels, 28, dim=2)
		labels = torch.repeat_interleave(labels, 28, dim=3)

		if data=='mnist':
			p_xm = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_init[~b_mask]),1)
		else:
			p_xm = td.Normal(loc = x_logits_init[~b_mask].reshape([-1,1]), scale =  torch.ones_like(b_data)[~b_mask].reshape([-1,1])) #.to(device,dtype = torch.float)

		#print(b_data_.shape, b_mask_.shape)
		b_data_[~b_mask_] = p_xm.sample([num_components]).reshape(-1)  
		#print(labels.shape, b_data_.shape)
		b_data_n = torch.cat((b_data_, labels.to(device,dtype = torch.float)), axis=1)
		#b_data_ = torch.Tensor.repeat(b_data_,[num_components,1,1,1])  
		out_encoder = encoder.forward(b_data_n)

		means[i,:,:] = out_encoder[...,:d]
		scales[i,:,:] = out_encoder[...,d:]

		#means[i,:,:] +=  (r2 - r1) * torch.rand( num_components, d) + r1 

	return logits, means, scales, logits_y


def init_params_labels(encoder, decoder, discriminative_model, p_z, b_data, b_mask, d, r1, r2, data='mnist'):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logits = torch.zeros(1,10) 
	means = torch.zeros(10, d)
	scales = torch.zeros(10, d)
	#Initialize parameters of q(z|y)

	for i in range(10):
		labels = torch.zeros(1, 10)
		labels[0,i] = 1.0
		labels = labels.reshape(1, 10, 1, 1)

		labels = torch.repeat_interleave(labels, 28, dim=2)
		labels = torch.repeat_interleave(labels, 28, dim=3)
		#print(labels.shape)
		b_data_ = torch.cat((b_data, labels.to(device,dtype = torch.float)), axis=1)
		out_encoder = encoder.forward(b_data_)

		means[i,:] = out_encoder[...,:d]
		scales[i,:] = out_encoder[...,d:]

	return logits, means, scales

##Not the updated function.
def init_mixture_labels(encoder, decoder, discriminative_model, p_z, b_data, b_mask, num_components, batch_size, d, r1, r2, data='mnist'):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	b_data_ = torch.Tensor.repeat(b_data,[num_components,1,1,1])  
	b_mask_ = torch.Tensor.repeat(b_mask,[num_components,1,1,1])
	x_logits_init = torch.zeros_like(b_data[:,0,:,:].reshape(1,1,28,28))

	#b_data_[~b_mask_] = p_xm.sample([num_components]).reshape(-1)  
	out_encoder = encoder.forward(b_data_)

	logits_y = discriminative_model.forward(b_data_[:,0,:,:].reshape(num_components,1,28,28))
	
	logits = torch.zeros(batch_size, num_components)
	means = torch.zeros(batch_size, num_components, d)
	scales = torch.ones(batch_size, num_components, d)
	#means = (r2 - r1) * torch.rand(batch_size, num_components, d) + r1
	#scales = (r2 - r1) * torch.rand(batch_size, num_components, d) + r1

	#print(out_encoder.shape)
	means[:,...] = out_encoder[...,:d]
	scales[:,...] = out_encoder[...,d:]

	sigma = torch.sum(scales.exp(),2)
	print(torch.sum(means,2))
	sigma = torch.Tensor.repeat(sigma.reshape(batch_size, num_components, 1), [1,1,d])

	means +=  (r2 - r1) * torch.rand(batch_size, num_components, d) + r1 
	#means +=  torch.mul(scales.exp(), (r2 - r1) * torch.rand(batch_size, num_components, d) + r1 )# Add random noise between [-1,1]
	#scales = -1*torch.ones(batch_size, num_components, d)

	print(torch.sum(means,2), torch.sum(scales.exp(),2))

	return logits, means, scales, logits_y

def init_mixture(encoder, decoder, p_z, b_data, b_mask, num_components, batch_size, d, r1=-1, r2=1, data='mnist', repeat= False):
	#rand = False
	#if rand:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#b_data = b_data.to(device,dtype = torch.float)
	#b_mask = b_mask.to(device,dtype = torch.bool)

	b_data_ = torch.Tensor.repeat(b_data,[num_components,1,1,1])  
	b_mask_ = torch.Tensor.repeat(b_mask,[num_components,1,1,1])
	x_logits_init = torch.zeros_like(b_data)

	if data=='mnist':
		p_xm = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_init[~b_mask]),1)
	else:
		channel_0 = torch.mean(b_data[:,0,:,:][b_mask[:,0,:,:]])
		channel_1 = torch.mean(b_data[:,1,:,:][b_mask[:,1,:,:]])
		channel_2 = torch.mean(b_data[:,2,:,:][b_mask[:,2,:,:]])
		channels_mean = torch.tensor([channel_0, channel_1, channel_2]).to(device,dtype = torch.float)        
		p_xm = td.Normal(loc = channels_mean.reshape([-1,1]), scale =  torch.ones_like(channels_mean).reshape([-1,1])) #.to(device,dtype = torch.float)
		mp = int(len(b_data[~b_mask])/3)
		print("missing pixels --", mp)       
		#p_xm = td.Normal(loc = 0.5 + x_logits_init[~b_mask].reshape([-1,1]), scale =  torch.ones_like(b_data)[~b_mask].reshape([-1,1])) #.to(device,dtype = torch.float)
		#b_data_[~b_mask] = 0.5
		b_data[:,0,:,:][~b_mask[:,0,:,:]] = channel_0
		b_data[:,1,:,:][~b_mask[:,1,:,:]] = channel_1
		b_data[:,2,:,:][~b_mask[:,2,:,:]] = channel_2
		b_data_ = torch.Tensor.repeat(b_data,[num_components,1,1,1])  
		#b_data_[~b_mask_] = p_xm.sample([num_components*mp]).reshape(-1) 

	out_encoder = encoder.forward(b_data_)

	logits = torch.zeros(batch_size, num_components)
	means = torch.zeros(batch_size, num_components, d)
	scales = torch.ones(batch_size, num_components, d)
	#means = (r2 - r1) * torch.rand(batch_size, num_components, d) + r1
	#scales = (r2 - r1) * torch.rand(batch_size, num_components, d) + r1

	#print(out_encoder.shape)
	means[:,...] = out_encoder[...,:d]
	scales[:,...] = out_encoder[...,d:]
	#print(means)
	sigma = torch.sum(scales.exp(),2)
	#print(torch.sum(means,2))
	sigma = torch.Tensor.repeat(sigma.reshape(batch_size, num_components, 1), [1,1,d])

	if data=='mnist':    
		means +=  (r2 - r1) * torch.rand(batch_size, num_components, d) + r1 
	else:
		means += (r2 - r1) * torch.rand(batch_size, num_components, d) + r1  #0.5 * torch.rand(batch_size, num_components, d)  #0.1 for not random case
	#means +=  torch.mul(scales.exp(), (r2 - r1) * torch.rand(batch_size, num_components, d) + r1 )# Add random noise between [-1,1]
	#scales = -1*torch.ones(batch_size, num_components, d)

	#print(torch.sum(means,2), torch.sum(scales.exp(),2))
	
	repeat = True
	if repeat == False:
		#Initializing missing pixels with 0
		#b_data[~b_mask] = p_xm.sample([]).reshape(-1)  
		#out_encoder = encoder.forward(b_data)
		#means[:, 0, :] = out_encoder[...,:d]
		#scales[:,0,:] = out_encoder[...,d:]

		b_data[~b_mask] = 0
		#Initializing missing pixels with KNN
		if data=='mnist':
			b_data[~b_mask] = k_neighbors_sample(b_data, b_mask) 
			out_encoder = encoder.forward(b_data)
			means[:, 1, :] = out_encoder[...,:d]
			scales[:,1,:] = out_encoder[...,d:]

			b_data[~b_mask] = 0
			
			x_logits_init, b_data_init, z_init = burn_in(b_data.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, burn_in_period=20)
			means[:, 2, :] = z_init[...,:d]
			scales[:,2,:] = z_init[...,d:]


			#b_data[~b_mask] = k_neighbors(b_data, b_mask).cuda() 
			#out_encoder = encoder.forward(b_data)
			#means[:, 2, :] = out_encoder[...,:d]
			#scales[:,2,:] = out_encoder[...,d:]
			#b_data[~b_mask] = 0
		else:
			b_data[~b_mask] = k_neighbors_sample_svhn(b_data, b_mask) 
			out_encoder = encoder.forward(b_data)
			means[:, 1, :] = out_encoder[...,:d]
			scales[:,1,:] = out_encoder[...,d:]

			#b_data[~b_mask] = 0
			#b_data[~b_mask] = k_neighbors_svhn(b_data, b_mask).cuda() 
			#out_encoder = encoder.forward(b_data)
			#means[:, 2, :] = out_encoder[...,:d]
			#scales[:,2,:] = out_encoder[...,d:]
			#b_data[~b_mask] = 0

	return logits, means, scales

def init_mixture_table(encoder, decoder, p_z, b_data, b_mask, num_components, batch_size, d, r1=-1, r2=1):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
	b_data_ = torch.Tensor.repeat(b_data,[num_components,1])  
	b_mask_ = torch.Tensor.repeat(b_mask,[num_components,1])
	x_logits_init = torch.zeros_like(b_data)

	out_encoder = encoder.forward(b_data_)

	logits = torch.zeros(batch_size, num_components)
	means = torch.zeros(batch_size, num_components, d)
	scales = torch.ones(batch_size, num_components, d)
	#means = (r2 - r1) * torch.rand(batch_size, num_components, d) + r1
	#scales = (r2 - r1) * torch.rand(batch_size, num_components, d) + r1

	#print(out_encoder.shape)
	means[:,...] = out_encoder[...,:d]
	scales[:,...] = out_encoder[...,d:]

	sigma = torch.sum(scales.exp(),2)
	#print(torch.sum(means,2))
	sigma = torch.Tensor.repeat(sigma.reshape(batch_size, num_components, 1), [1,1,d])

	means +=  (r2 - r1) * torch.rand(batch_size, num_components, d) + r1 

	#means +=  torch.mul(scales.exp(), (r2 - r1) * torch.rand(batch_size, num_components, d) + r1 )# Add random noise between [-1,1]
	#scales = -1*torch.ones(batch_size, num_components, d)

	#print(torch.sum(means,2), torch.sum(scales.exp(),2))
	
	return logits, means, scales




