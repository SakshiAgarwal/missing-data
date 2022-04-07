import torch
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
"""
Initialize Hyperparameters
"""
d = 50 #latent dim
batch_size = 64
learning_rate = 1e-3
num_epochs = 0
stop_early=False
binary_data = False
K=1
valid_size = 0.1
num_epochs_test = 1000
results="/home/sakshia1/myresearch/missing_data/miwae/pytorch/results/mnist-" + str(binary_data) + "-"
ENCODER_PATH = "models/e_model_"+ str(binary_data) + ".pt"
DECODER_PATH = "models/d_model_"+ str(binary_data) + ".pt"

"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""
train_loader, val_loader = train_valid_loader(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data)

"""
Initialize the network and the Adam optimizer
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder =  FlatWideResNet(channels=1, size=1, levels=3, blocks_per_level=2, out_features = 2*d, shape=(28,28))
decoder = FlatWideResNetUpscaling(channels=1, size=1, levels=3, blocks_per_level=2, in_features = d, shape=(28,28))
encoder = encoder.cuda()
decoder = decoder.cuda()

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

best_loss, count = 0, 0

encoder, decoder = train_VAE(num_epochs, train_loader, val_loader, ENCODER_PATH, DECODER_PATH, results, encoder, decoder, optimizer, p_z, device, d, stop_early)

### Load model 
checkpoint = torch.load(ENCODER_PATH)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH)
decoder.load_state_dict(checkpoint['model_state_dict'])

for i in [0.5, 0.2, 0.7, 0.9]:
    ### Get test loader for different missing percentage value
    test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = i),batch_size=1)

    test_log_likelihood, test_loss, test_mse, nb, = 0, 0, 0, 0
    
    start = datetime.now()

    ### Optimize over test variational parameters
    for data in test_loader:
        nb += 1
        b_data, b_mask, b_full = data
        missing_counts = 1*28*28 - int(b_mask.sum())

        ## For each image get params
        #xm_params = torch.zeros(missing_counts)
        xm_params = torch.rand(missing_counts)
        xm_params = xm_params.to(device)
        xm_params.requires_grad = True
        
        b_data = b_data.to(device,dtype = torch.float)
        b_mask = b_mask.to(device,dtype = torch.bool)

        p_xm = td.Independent(td.ContinuousBernoulli(logits=(xm_params).cuda()),1)
        #p_xm = td.ContinuousBernoulli(logits=(xm_params).cuda())

        #test_optimizer = torch.optim.LBFGS([xm_params]) ##change, sgd, adagrad
        test_optimizer = torch.optim.Adam([xm_params], lr=0.01, betas=(0.9, 0.999)) 

        for k in range(num_epochs_test):
            test_optimizer.zero_grad()
            loss, loglike = xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, p_xm = p_xm,encoder = encoder, decoder = decoder, d=d, K=2000)
            loss.backward(retain_graph=True) 
            test_optimizer.step()
            #test_optimizer.step(lambda: xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, p_xm = p_xm,encoder = encoder, decoder = decoder, d=d, K=1)[0]) # when using LBFGS as optim
            xhat = xm_impute(iota_x = b_data,mask = b_mask,p_z = p_z, p_xm= p_xm, encoder = encoder,decoder = decoder, d=d, L=1).cpu().data.numpy().reshape(1,1,28,28)

            if binary_data:
                xhat[xhat < 0.5] = 0
                xhat[xhat >= 0.5] = 1

            ### Get mse error on imputation
            err = np.array([mse(xhat,b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
            print("Loss : {}".format(loss))
            #print(k,err, loss)
            
            if k != 0 and k%100==0:
                img = 255*b_full.cpu().numpy()
                plot_image(np.squeeze(img),results + str(i) + "image1/epoch_" + str(k) + "xm-true.png" )
                
                b_data[~b_mask] = torch.sigmoid(xm_params)
                img = 255*b_data.cpu().data.numpy()
                plot_image(np.squeeze(img),results + str(i) + "image1/epoch_" +   str(k) + "xm-input.png" )

                xhat = 255*xm_impute(iota_x = b_data,mask = b_mask,p_z=p_z, p_xm=p_xm, encoder = encoder,decoder = decoder, d=d, L=1).cpu().data.numpy().reshape(1,1,28,28)               

                mask = b_mask.cpu().data.numpy().astype(np.bool)

                if binary_data:
                    xhat[xhat < 0.5] = 0
                    xhat[xhat >= 0.5] = 1

                plot_image(np.squeeze(xhat), results + str(i) + "image1/epoch_" + str(k) + "xm-imputed.png" )

            if (k+1)%2000==0:
                exit()

        test_mse += err 
        test_log_likelihood += loglike
        test_loss += loss
        
        gc.collect()
        print('Test-image {} Test Loss {}, log-likelihood {}, test-mse {}'.format(nb, test_loss, test_log_likelihood, test_mse))
        if (nb>=1):
            break

    end = datetime.now()
    diff = end - start

    ### Save test log-likelihood
    #with open(results + "test.txt", 'a') as f:
    #        f.write("xm nb " + str(num_epochs_test) + "% missing  " + str(i) + "\t test_log_likelihood/img " +  str(float(test_log_likelihood/(nb))) + " \t test_log_likelihood/pixel" +  str(float(test_log_likelihood/(nb*(1-i)*28*28))) + "\t test_mse/img " + str(test_mse/(nb)) + "\t test_mse/pixel " + str(test_mse/(nb*i*28*28)) + "\t time " + str(diff.total_seconds()/60) + "\n")

    test_log_likelihood, test_loss, test_mse = test_log_likelihood/(nb*(1-i)*28*28), test_loss/(nb), test_mse/(nb*i*28*28)

    z=0
    if z:
        ### Sample and plot one image
        for data in random.sample(list(test_loader), 1):
            b_data, b_mask, b_full = data
            b_data = b_data.to(device,dtype = torch.float)
            b_mask = b_mask.to(device,dtype = torch.float)
                    
            count +=1
            
            img = 255*b_full.cpu().numpy()
            plot_image(np.squeeze(img),results + str(i) + str(count) + "xm-true.png" )
            
            img = b_data.cpu().numpy()
            plot_image(np.squeeze(img),results + str(i) + str(count) + "xm-true-masked.png" )
            
            mask = b_mask.cpu().data.numpy().astype(np.bool)
            xhat = xm_impute(iota_x = b_data,mask = b_mask,p_z=p_z, p_xm=p_xm, encoder = encoder,decoder = decoder, d=d, L=1).cpu().data.numpy().reshape(1,1,28,28)  

            img = b_data.cpu().numpy()
            plot_image(np.squeeze(img),results + str(i) + str(count) + "xm-input.png" )
            
            if binary_data:
                xhat[xhat < 0.5] = 0
                xhat[xhat >= 0.5] = 1

            plot_image(np.squeeze(xhat), results + str(i) + str(count) + "xm-imputed.png" )

