import torch
torch.cuda.set_device(2)
print(torch.cuda.current_device())
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
#from networks import simple_encoder, simple_decoder
import gc

"""
Initialize Hyperparameters
"""

d = 50 #latent dim, 
batch_size = 64 
learning_rate = 1e-3
num_epochs = 100
stop_early = False
binary_data = False
K=1
valid_size = 0.1
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

exit()
checkpoint = torch.load(ENCODER_PATH)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH)
decoder.load_state_dict(checkpoint['model_state_dict'])

example = np.zeros((1,28,28))
plot_image(np.squeeze(example), results +  "example.png" )
example = np.random.rand(1,28,28)
print(example)
plot_image(np.squeeze(example), results + "example2.png" )

for perc_miss in [0.7, 0.2, 0.5, 0.7, 0.9]:
    ### Get test loader for different missing percentage value
    test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = perc_miss),batch_size=1)
      
    print(perc_miss)
    with torch.no_grad():
        ### Load model if stopped early
        test_log_likelihood, test_loss, test_mse, nb, = 0.0, 0.0, 0.0, 0
        ### Calculate test-loglikelihood

        count=0
        for data in test_loader:
        #for data in random.sample(list(test_loader), 1):
                b_data, b_mask, b_full = data
                b_data = b_data.to(device,dtype = torch.float)
                b_mask = b_mask.to(device,dtype = torch.bool)
                mask = b_mask.cpu().data.numpy().astype(np.bool)

                #print(np.shape(mask), b_data.shape)
                count +=1
                
                img = b_full.cpu().data.numpy()         ##Added .data to convert from tensor
                plot_image(np.squeeze(img),results + str(perc_miss) + str(count) + "true.png" )

                #print("missing ", perc_miss)
                #print(int(b_mask.sum()))
                img_ = b_data.cpu().data.numpy()
                plot_image(np.squeeze(img_),results + str(perc_miss) + str(count) + "true-masked.png" )

                xhat = b_data.cpu().data.numpy()
                xhat[~mask] = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).cpu().data.numpy().reshape(1,1,28,28)[~mask]

                #xhat[~mask] = miwae_impute(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),L=10).cpu().data.numpy()[~mask]

                plot_image(np.squeeze(xhat), results + str(perc_miss) + str(count) + "imputed.png" )

                if binary_data:
                    xhat[xhat < 0.5] = 0
                    xhat[xhat >= 0.5] = 1
                    
                xhat[xhat < 0.5] = 0
                xhat[xhat >= 0.5] = 1
                
                ### Get mse error on imputation
                #err = np.array([mse(xhat,b_data_full.cpu().data.numpy(),mask)])
                #mse_train = np.append(mse_train,err,axis=0)
                #test_mse += err 

                plot_image(np.squeeze(xhat), results + str(perc_miss) + str(count) + "binary-imputed.png" )
                nb+=1
                break


        '''
        for data in test_loader:
            nb += 1
            b_data, b_mask, b_full = data
            b_data = b_data.to(device,dtype = torch.float)
            b_mask = b_mask.to(device,dtype = torch.float)
            loss, loglike = mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder, decoder = decoder, p_z = p_z, d=d, K=1)

            xhat = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z =p_z, d=d, L=1).cpu().data.numpy().reshape(1,1,28,28)
            if binary_data:
                xhat[xhat < 0.5] = 0
                xhat[xhat >= 0.5] = 1

            ### Get mse error on imputation
            b_mask = b_mask.cpu().data.numpy().astype(bool)
            err = np.array([mse(xhat,b_full.cpu().data.numpy(),b_mask)])
            #se_train = np.append(mse_train,err,axis=0)

            test_mse += float(err)
            test_log_likelihood += float(loglike)
            test_loss += float(loss)

            if nb>=100:
                break
        '''

        ### Save test log-likelihood
        with open(results + "test.txt", 'a') as f:
            f.write("% missing " + str(perc_miss) + "\t test_log_likelihood/img " +  str(float(test_log_likelihood/(nb))) + " \t test_log_likelihood/pixel" +  str(float(test_log_likelihood/(nb*(1-perc_miss)*28*28))) + "\t test_mse/img " + str(test_mse/(nb)) + "\t test_mse/pixel " + str(test_mse/(nb*perc_miss*28*28)) + "\n")

        test_log_likelihood, test_loss, test_mse = test_log_likelihood/(nb*(1-perc_miss)*28*28), test_loss/(nb*(1-perc_miss)*28*28), test_mse/(nb*perc_miss*28*28)

        print(' Test Loss/pixel {}, log-likelihood/pixel {}'.format(test_loss, test_log_likelihood))

        ### Sample and plot one image



#def main():
#    args = sys.argv[1:]
#    print(args)
#    exit()

