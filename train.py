import torch
import numpy as np
from loss import *
import gc

def train_VAE(num_epochs, train_loader, val_loader, ENCODER_PATH, DECODER_PATH, results, encoder, decoder, optimizer, p_z, device, d, stop_early):
	print("Training ---")
	
	for epoch in range(num_epochs):
	    train_loss = 0
	    train_log_likelihood = 0
	    nb = 0
	    train_mse = 0
	    for data in train_loader:
	        nb +=1
	        b_data, b_mask = data
	        batch_size = b_data.shape[0]
	        #print(b_data.dtype)
	        b_data = b_data.to(device,dtype = torch.float)
	        b_mask = b_mask.to(device,dtype = torch.float)
	        # calculate mvae loss
	        batch_size = b_data.shape[0]
	        loss, loglike = mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)
	        train_log_likelihood += loglike
	        train_loss += loss 
	        # Backpropagation based on the loss
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	        #print('Epoch {}: Loss {}'.format(epoch, loss))

	       	xhat = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).cpu().data.numpy().reshape(batch_size,1,28,28)

	       	b_mask = b_mask.cpu().data.numpy().astype(bool)
	       	#b_mask = 1
	       	#b_mask[b_data>0] = 0

	       	err = np.array([mse(xhat,b_data.cpu().data.numpy(),~b_mask)])

	       	train_mse += err
	       	#print(err)

	    gc.collect()

	    train_loss = train_loss/nb
	    train_log_likelihood = train_log_likelihood/nb 
	    print('Epoch {}: Training : Loss {}, log-likelihood {}'.format(epoch, train_loss, train_log_likelihood))


	    with open(results + ".txt", 'a') as f:
	        #f.write(str(-train_loss.cpu().data.numpy().astype(np.float)) + " \t " + str(train_log_likelihood.cpu().data.numpy().astype(np.float)) + "\t")
	        f.write("Train-loglikelihood/pixel " + str((train_log_likelihood/(batch_size*28*28)).cpu().data.numpy().astype(np.float)) + " \t Train-log-likelihood/batch" + str(train_log_likelihood.cpu().data.numpy().astype(np.float)) + "\t")
	    #print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda()).cpu().data.numpy())) # Gradient step

	    ##### Get error on Validation set 
	    val_log_likelihood, val_loss, nb = 0, 0, 0
	   
	    for data in val_loader:
	        nb += 1           
	        #print(nb)
	        b_data_val, b_mask_val = data
	        b_data_val = b_data_val.to(device,dtype = torch.float)
	        b_mask_val = b_mask_val.to(device,dtype = torch.float)

	        batch_size = b_data_val.shape[0]
	        #print(batch_size,b_data_val[0])
	        loss, loglike = mvae_loss(iota_x = b_data_val,mask = b_mask_val,encoder = encoder, decoder = decoder, p_z= p_z, d=d, K=1)
	        val_log_likelihood += loglike
	        val_loss += loss

	    val_log_likelihood, val_loss = val_log_likelihood/nb, val_loss/nb
	    print(' Validation Loss {}, log-likelihood {}'.format( val_loss, val_log_likelihood))

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
		    torch.save({'model_state_dict': decoder.state_dict()}, DECODER_PATH)

	    with open(results + ".txt", 'a') as f:
	        ### SAVE ELBO (-loss) and likelihood
	        f.write(str((val_log_likelihood/(batch_size*28*28)).cpu().data.numpy().astype(np.float)) + " \t " + str(val_log_likelihood.cpu().data.numpy().astype(np.float)) + "\n")

	return encoder, decoder
