import torch
import numpy as np
from loss import *
import gc
from torch.nn import functional as F
import torch.nn as nn


def weights_init(layer):
  if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
   

def train_VAE_uci(num_epochs, trainset, validset, ENCODER_PATH,  results, encoder, decoder, optimizer, p_z, device, d, DECODER_PATH=None, log_sigma=None):
	encoder.apply(weights_init)
	decoder.apply(weights_init)
	n = len(trainset)
	p = trainset.shape[1]

	n_val = validset.shape[0]
	bs=64
	K=1
	print("trainset size: ", n,p)
	mask = np.ones((n,p))
	mask_val = np.ones((n_val,p))

	for epoch in range(num_epochs):
		train_loss = 0
		train_log_likelihood = 0
		nb = 0
		train_mse = 0

		perm = np.random.permutation(n)
		batches_data = np.array_split(trainset[perm,], n/bs)
		batches_mask = np.array_split(mask[perm,], n/bs)

		perm = np.random.permutation(n_val)
		batches_data_val = np.array_split(validset[perm,], n_val/bs)
		batches_mask_val = np.array_split(mask_val[perm,], n_val/bs)

		for it in range(len(batches_data)):
			#print(log_sigma)
			optimizer.zero_grad()
			encoder.zero_grad()
			decoder.zero_grad()
			b_data = torch.from_numpy(batches_data[it]).float().cuda()
			b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
			loss = miwae_loss(b_data,b_mask, encoder, decoder, d, K, p_z)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()

		print("training loss : ", train_loss)

		val_loss = 0
		for it in range(len(batches_data_val)):
			#print(log_sigma)
			b_data_val = torch.from_numpy(batches_data_val[it]).float().cuda()
			b_mask_val = torch.from_numpy(batches_mask_val[it]).float().cuda()
			loss = miwae_loss(b_data_val,b_mask_val, encoder, decoder, d, K, p_z)
			val_loss += loss.item()
		
		print("validation loss : ", val_loss)

	torch.save({'model_state_dict': encoder.state_dict()}, ENCODER_PATH)
	if DECODER_PATH is not None:
		torch.save({'model_state_dict': decoder.state_dict()}, DECODER_PATH)

	return encoder, decoder



def train_VAE(num_epochs, train_loader, val_loader, ENCODER_PATH,  results, encoder, decoder, optimizer, p_z, device, d, stop_early, with_labels=False, DECODER_PATH=None):
	print("Training ---")
	torch.cuda.empty_cache()
	for epoch in range(num_epochs):
		train_loss = 0
		train_log_likelihood = 0
		nb = 0
		train_mse = 0
		for data in train_loader:
			nb +=1
			b_data, b_mask, b_full, labels = data

			#print(labels.shape)
			batch_size = b_data.shape[0]
			#print(b_data.dtype)
			b_data = b_data.to(device,dtype = torch.float)
			b_mask = b_mask.to(device,dtype = torch.float)
			labels = labels.to(device,dtype = torch.float)

			# calculate mvae loss
			batch_size = b_data.shape[0]

			loss, loglike = mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1, with_labels=with_labels, labels= labels)
			# Backpropagation based on the loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#print('Epoch {}: Loss {}'.format(epoch, loss))

			train_log_likelihood += float(loglike)
			train_loss += float(loss)

			xhat = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1, with_labels=with_labels, labels= labels)[0].cpu().data.numpy().reshape(batch_size,1,28,28)
			b_mask = b_mask.cpu().data.numpy().astype(bool)

			#b_mask = 1
			#b_mask[b_data>0] = 0
			#print(b_mask.shape, xhat.shape, b_data[:,0,:,:].shape )
			err = np.array([mse(xhat,b_data[:,0,:,:].cpu().data.numpy().reshape(batch_size,1,28,28),~b_mask)])
			train_mse += float(err)
				#print(err)

		gc.collect()

		train_loss = train_loss/nb
		train_log_likelihood = train_log_likelihood/nb 
		print('Epoch {}: Training : Loss {}, log-likelihood {}'.format(epoch, train_loss, train_log_likelihood))

		#with open(results + ".txt", 'a') as f:
	    #f.write(str(-train_loss.cpu().data.numpy().astype(np.float)) + " \t " + str(train_log_likelihood.cpu().data.numpy().astype(np.float)) + "\t")
		#f.write("Train-loglikelihood/pixel " + str((train_log_likelihood/(batch_size*28*28))) + " \t Train-log-likelihood/batch" + str(train_log_likelihood) + "\t")
	
		#print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda()).cpu().data.numpy())) # Gradient step

		##### Get error on Validation set 
		val_log_likelihood, val_loss, nb = 0, 0, 0

		for data in val_loader:
			with torch.no_grad():
				nb += 1           
				#print(nb)
				b_data_val, b_mask_val, b_full, labels = data
				b_data_val = b_data_val.to(device,dtype = torch.float)
				b_mask_val = b_mask_val.to(device,dtype = torch.float)
				labels = labels.to(device,dtype = torch.float)

				batch_size = b_data_val.shape[0]
				#print(batch_size,b_data_val[0])
				loss, loglike = mvae_loss(iota_x = b_data_val,mask = b_mask_val,encoder = encoder, decoder = decoder, p_z= p_z, d=d, K=1, with_labels=with_labels, labels= labels)
				val_log_likelihood += float(loglike)
				val_loss += float(loss)

		val_log_likelihood, val_loss = val_log_likelihood/nb, val_loss/nb
		print(' Validation Loss {}, log-likelihood {}'.format( val_loss, val_log_likelihood))

		###### If there is a stop condition
		if(stop_early):
			if(best_loss>val_loss):
				best_loss = val_loss
				torch.save({'model_state_dict': encoder.state_dict()}, ENCODER_PATH)
				if DECODER_PATH is not None:
					torch.save({'model_state_dict': decoder.state_dict()}, DECODER_PATH)
			else:
				count += 1
				if(count>2):
					break
		else:
			torch.save({'model_state_dict': encoder.state_dict()}, ENCODER_PATH)
			if DECODER_PATH is not None:
				torch.save({'model_state_dict': decoder.state_dict()}, DECODER_PATH)

		#with open(results + ".txt", 'a') as f:
		### SAVE ELBO (-loss) and likelihood
		#f.write(str((val_log_likelihood/(batch_size*28*28))) + " \t " + str(val_log_likelihood) + "\n")

	print("Memory allocated after training ---")
	torch.cuda.empty_cache()

	return encoder, decoder

def train_pygivenx(num_epochs, train_loader, val_loader, ENCODER_PATH, results, encoder, optimizer, device, d, stop_early):
	print("Training ---")

	torch.cuda.empty_cache()

	CEloss = torch.nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		print(epoch)
		train_loss = 0
		train_log_likelihood = 0
		nb = 0
		train_mse = 0
		for data in train_loader:
			nb +=1
			b_data, b_mask, labels = data

			#print(labels.shape)
			batch_size = b_data.shape[0]
			#print(b_data.dtype)
			b_data = b_data.to(device,dtype = torch.float)
			b_mask = b_mask.to(device,dtype = torch.float)
			labels = labels.to(device,dtype = torch.float)
			# calculate classification loss
			batch_size = b_data.shape[0]
			out_encoder = encoder.forward(b_data)
			#out_probs = torch.nn.functional.softmax(out_encoder, dim=1)

			loss = CEloss(out_encoder, labels)
			#print(loss)
			# Backpropagation based on the loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#train_log_likelihood += float(loglike)
			train_loss += float(loss)

			b_mask = b_mask.cpu().data.numpy().astype(bool)
		gc.collect()

		train_loss = train_loss/nb
		print('Epoch {}: Training : Loss {} '.format(epoch, train_loss))

		##### Get error on Validation set 
		val_log_likelihood, val_loss, nb = 0, 0, 0

		for data in val_loader:
			with torch.no_grad():
				nb += 1           
				#print(nb)
				b_data_val, b_mask_val, labels = data
				b_data_val = b_data_val.to(device,dtype = torch.float)
				b_mask_val = b_mask_val.to(device,dtype = torch.float)
				labels = labels.to(device,dtype = torch.float)

				batch_size = b_data_val.shape[0]
				out_encoder = encoder.forward(b_data_val)
				#out_probs = torch.nn.functional.softmax(out_encoder, dim=1)
				loss = CEloss(out_encoder, labels)
				val_loss += float(loss)

		val_loss = val_loss/nb
		print(' Validation Loss {}'.format( val_loss))

		###### If there is a stop condition
		if(stop_early):
			if(best_loss>val_loss):
				best_loss = val_loss
				torch.save({'model_state_dict': encoder.state_dict()}, ENCODER_PATH)
				torch.save({'model_state_dict': decoder.state_dict()}, DECODER_PATH)
			else:
				count += 1
				if(count>2):
					break
		else:
			torch.save({'model_state_dict': encoder.state_dict()}, ENCODER_PATH)

	print("Memory allocated after training ---")
	torch.cuda.empty_cache()

	return encoder


def train_VAE_SVHN(num_epochs, train_loader, val_loader, ENCODER_PATH,  results, encoder, decoder, optimizer, p_z, device, d, stop_early, annealing=False, DECODER_PATH= None):
	print("Training ---")

	torch.cuda.empty_cache()

	best_loss = 2000000
	stop_early = True

	step = 0
	k_l_train = []
	loglike_train = []
	loss_train = []
	for epoch in range(num_epochs):
		train_loss = 0
		train_log_likelihood = 0
		nb = 0
		train_mse = 0
		k_l_epoch = 0
		loglike_epoch = 0

		for data in train_loader:
			nb +=1
			step += 1
			alpha = min(step / 500., 10.)
			b_data, b_mask = data
			batch_size = b_data.shape[0]
			#print(b_data.dtype)
			b_data = b_data.to(device,dtype = torch.float)
			b_mask = b_mask.to(device,dtype = torch.float)
			# calculate mvae loss
			batch_size = b_data.shape[0]

			if annealing:
				loss, _ , k_l, loglike  = mvae_loss_svhn(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1, alpha=alpha)
			else:
				loss, _ , k_l, loglike  = mvae_loss_svhn(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1, alpha=1)

			# Backpropagation based on the loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loglike_epoch += float(loglike)
			k_l_epoch += float(k_l)
			train_log_likelihood += float(loglike)
			train_loss += float(loss)

			#xhat = mvae_impute_svhn(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder,  p_z = p_z, d=d, L=1)[0].cpu().data.numpy().reshape(batch_size,3,32,32)

			#b_mask = b_mask.cpu().data.numpy().astype(bool)
			#b_mask = 1
			#b_mask[b_data>0] = 0
			#err = np.array([mse(xhat,b_data.cpu().data.numpy(),~b_mask)])

			#train_mse += float(err)
				#print(err)

		k_l_train.append(k_l_epoch/nb)
		loss_train.append(train_loss/nb)
		loglike_train.append(loglike_epoch/nb)

		print(step)
		gc.collect()

		train_loss = train_loss/nb
		train_log_likelihood = train_log_likelihood/nb 
		print('Epoch {}: Training : Loss {}, log-likelihood {}'.format(epoch, train_loss, train_log_likelihood))


		#with open(results + ".txt", 'a') as f:
		#f.write(str(-train_loss.cpu().data.numpy().astype(np.float)) + " \t " + str(train_log_likelihood.cpu().data.numpy().astype(np.float)) + "\t")
		#	f.write("Train-loglikelihood/pixel " + str((train_log_likelihood/(batch_size*32*32))) + " \t Train-log-likelihood/batch" + str(train_log_likelihood) + "\t")
		
		#print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda()).cpu().data.numpy())) # Gradient step

		##### Get error on Validation set 
		val_log_likelihood, val_loss, nb = 0, 0, 0

		for data in val_loader:
			with torch.no_grad():
				nb += 1           
				#print(nb)
				b_data_val, b_mask_val = data
				b_data_val = b_data_val.to(device,dtype = torch.float)
				b_mask_val = b_mask_val.to(device,dtype = torch.float)
				batch_size = b_data_val.shape[0]

				if annealing:
					loss, _ , k_l, loglike  = mvae_loss_svhn(iota_x = b_data_val,mask = b_mask_val,encoder = encoder, decoder = decoder, p_z= p_z, d=d, K=1, alpha=alpha)
				else:
					loss, _ , k_l, loglike  = mvae_loss_svhn(iota_x = b_data_val,mask = b_mask_val,encoder = encoder, decoder = decoder, p_z= p_z, d=d, K=1, alpha=1)

				val_log_likelihood += float(loglike)
				val_loss += float(loss)

		val_log_likelihood, val_loss = val_log_likelihood/nb, val_loss/nb
		print(' Validation Loss {}, log-likelihood {}'.format( val_loss, val_log_likelihood))

		###### If there is a stop condition
		if(stop_early):
			if(best_loss>val_loss):
				best_loss = val_loss
				torch.save({'model_state_dict': encoder.state_dict()}, ENCODER_PATH)
				torch.save({'model_state_dict': decoder.state_dict()}, DECODER_PATH)
				print("model saved at epoch :", epoch)
		else:
			torch.save({'model_state_dict': encoder.state_dict()}, ENCODER_PATH)
			torch.save({'model_state_dict': decoder.state_dict()}, DECODER_PATH)
#			torch.save({'model_state_dict': sigma_decoder.state_dict()}, DECODER_PATH)

	#x = np.arange(10)
	#plt.plot(x, k_l_train, color='red', label="kl")
	#plt.plot(x, loss_train, color='black', label="loss")
	#plt.plot(x, loglike_train, color='orange', label="p(x|z)")

	print("Memory allocated after training ---")
	torch.cuda.empty_cache()

	return encoder, decoder

