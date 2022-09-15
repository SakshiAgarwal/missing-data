import os
from numba import cuda
cuda.select_device(3)
print(cuda.current_context().get_memory_info())
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["NVIDIA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import torch
torch.cuda.set_device(3)
print(torch.cuda.current_device())
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
from resnets import WideResNet, WideResNetUpscaling, FlatWideResNet, FlatWideResNetUpscaling
import os
from data import *
from loss import *
from train import *
from plot import *
from datetime import datetime
import gc
from inference import m_g_sampler
"""
Initialize Hyperparameters
"""
d = 50 #latent dim
batch_size = 64
learning_rate = 1e-3
num_epochs = 0
stop_early= False
binary_data = False
K=1
valid_size = 0.1
num_epochs_test = 50
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
print(device)

encoder =  FlatWideResNet(channels=1, size=1, levels=3, blocks_per_level=2, out_features = 2*d, shape=(28,28))
decoder = FlatWideResNetUpscaling(channels=1, size=1, levels=3, blocks_per_level=2, in_features = d, shape=(28,28))
encoder = encoder.cuda()
decoder = decoder.cuda()
print(torch.cuda.current_device())

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

best_loss, count = 0, 0

if num_epochs>0:
    encoder, decoder = train_VAE(num_epochs, train_loader, val_loader, ENCODER_PATH, DECODER_PATH, results, encoder, decoder, optimizer, p_z, device, d, stop_early)

### Load model 
checkpoint = torch.load(ENCODER_PATH)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH)
decoder.load_state_dict(checkpoint['model_state_dict'])
print(torch.cuda.current_device())

print("model loaded")
for params in encoder.parameters():
    params.requires_grad = False

for params in decoder.parameters():
    params.requires_grad = False

for i in [0.5]:
    ### Get test loader for different missing percentage value
    print("memory before --" )
    print(torch.cuda.memory_allocated(device=3))
    test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = i),batch_size=1)

    print("test data loaded")
    test_log_likelihood, test_loss, test_mse, nb, = 0, 0, 0, 0
    
    start = datetime.now()

    np.random.seed(5678)
    ### Optimize over test variational parameters

   
    for data in test_loader:
        nb += 1
        b_data, b_mask, b_full = data

        missing_pattern_x = np.argwhere(b_mask == False)[2]
        missing_pattern_y = np.argwhere(b_mask == False)[3]

        missing_counts = 1*28*28 - int(b_mask.sum())

        k_plots = 50
        check_plots = np.random.choice(missing_counts, k_plots, replace=False)
        #check_plots_y = np.random.choice(missing_counts, k_plots, replace=False)

        check_plots = np.asarray(check_plots).astype(np.int32)
        #check_plots_y = np.asarray(check_plots_y).astype(np.int32)
        missing_pattern_x = missing_pattern_x[check_plots]
        missing_pattern_y = missing_pattern_y[check_plots]

        #print(missing_pattern_x, missing_pattern_y)

        params_plots = np.zeros((k_plots, num_epochs_test))

        b_full = b_full.reshape(28,28)
        true_pixels = [b_full[missing_pattern_x[k_plot],missing_pattern_y[k_plot]].item()  for k_plot in range(k_plots)]

        print(true_pixels, missing_pattern_x, missing_pattern_y)
        ## For each image get params
        #xm_params = torch.zeros(missing_counts)
        #xm_params = torch.rand(missing_counts)
        
        #xm_params is a tensor now...
        b_data = b_data.to(device,dtype = torch.float)
        b_mask = b_mask.to(device,dtype = torch.bool)

         ##Init xm_params with output of decoder:
        pseudo_gibbs_loss = []
        pseudo_gibbs_mse = []

        #p_xm = td.ContinuousBernoulli(logits=(xm_params).cuda())

        b_data[~b_mask] = 0
        b_full = b_full.reshape([1,1,28,28]).to(device,dtype = torch.float)
        lower_bound = - mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)[0].item()
        upper_bound = - mvae_loss(iota_x = b_full,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)[0].item()

        upper_imp = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)
        v = torch.sigmoid(upper_imp)
        imputation = b_data
        imputation[~b_mask] = v[~b_mask]
        imputation = imputation.cpu().data.numpy().reshape(28,28)

        upper_err = np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])

        lower_imp = mvae_impute(iota_x = b_full,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)
        v = torch.sigmoid(lower_imp)
        imputation = b_data
        imputation[~b_mask] = v[~b_mask]
        imputation = imputation.cpu().data.numpy().reshape(28,28)

        lower_err = np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])

        burn_in_period = 50

        b_data[~b_mask] = 0
        # Burning in samples
        for l in range(burn_in_period):
            x_logits = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)
            b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
            #torch.sigmoid(x_logits[~b_mask])

        x_logits_burn = x_logits
        
        # Pseudo-gibbs sampler 
        for l in range(num_epochs_test):
            loss, loglike = mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)
            x_logits = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)

            b_data[~b_mask] = torch.sigmoid(x_logits[~b_mask])
            err = np.array([mse(b_data.cpu().data.numpy(),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy())])
            pseudo_gibbs_loss.append(-loss.item())
            print(-loss.item())
            pseudo_gibbs_mse.append(err)        
            b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
   

        #Imput image with pseudo-gibbs
        b_data[~b_mask] = torch.sigmoid(x_logits[~b_mask])
        plot_image(np.squeeze(b_data.cpu().data.numpy()), results + str(i) + "image1/" + "pseudo-gibbs.png" )

        ##M-with-gibbs sampler
        x_logits = x_logits_burn
        b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
        m_loglikelihood, m_error, x_full_logits =  m_g_sampler(iota_x = b_data, full = b_full, mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1, T=num_epochs_test)

        #Imput image with metropolis-within-pseudo-gibbs
        b_data[~b_mask] = torch.sigmoid(x_full_logits[~b_mask])
        plot_image(np.squeeze(b_data.cpu().data.numpy()), results + str(i) + "image1/" + "metropolis-within-pseudo-gibbs.png" )

        ## Clamp with lower prob=0.05, max prob = 0.95
        #xm_params = torch.clamp(xm_params, min=-2.94, max=2.94)
        #min p = 0.01

        xm_params = x_logits_burn[~b_mask] 
        print(xm_params)
        xm_params = torch.clamp(xm_params, min=-20, max=20)
        xm_params = xm_params.to(device)

        print(xm_params.tolist())

        upper_prob = torch.tensor(1, dtype=torch.float64) - torch.tensor(float(1.8)*np.exp(-70), dtype=torch.float64)
        lower_prob = torch.tensor(float(1.8)*np.exp(-70), dtype=torch.float64)
        print(upper_prob, lower_prob)
        print("***********************")

        p_xm = td.Independent(td.ContinuousBernoulli(logits=(xm_params).cuda()),1)       ##changed to probs, since sigmoid outputs     
        #p_xm = td.Independent(td.ContinuousBernoulli(probs=(upper_prob*torch.sigmoid(xm_params)+lower_prob).cuda()),1) 
        image = b_data
        a = p_xm.rsample()
        #print(a.tolist(),a.shape)
        image[~b_mask] = a.reshape(-1)   
        #print(image[~b_mask].tolist())
        plot_image(np.squeeze(image.cpu().data.numpy()), results + str(i) + "image1/epoch_0" + "xm-initial-sampled.png" )

        xm_params.requires_grad = True
        #test_optimizer = torch.optim.LBFGS([xm_params]) ##change, sgd, adagrad
        test_optimizer = torch.optim.Adam([xm_params], lr=10.0, betas=(0.9, 0.999)) 
        #test_optimizer = torch.optim.Adagrad([xm_params], lr=0.1) 
        xm_params_constrained = torch.clamp(xm_params, min=-100, max=100)
        #xm_params_constrained.requires_grad = False
        loss_acc = []

        xm_loglikelihood = []
        xm_mse = []
        xm_approx_loss = []
        for k in range(num_epochs_test):
            xm_params_constrained = torch.clamp(xm_params, min=-80, max=80)

            xm_params_constrained.detach()
            p_xm = td.Independent(td.ContinuousBernoulli(logits=(xm_params_constrained).cuda()),1)
            #p_xm = td.Independent(td.ContinuousBernoulli(probs=(upper_prob*torch.sigmoid(xm_params)+lower_prob).cuda()),1) 

            print(xm_params_constrained.tolist())
            #print(cuda.current_context().get_memory_info())
            test_optimizer.zero_grad()
            loss, approx_loss = xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, p_xm = p_xm,encoder = encoder, decoder = decoder, d=d, K=1, K_z= 1)
            loss.backward()            

            test_optimizer.step()

            imputation = b_data
            v = torch.sigmoid(xm_params).detach()
            imputation[~b_mask] = v
            imputation = imputation.cpu().data.numpy().reshape(28,28)
            #imputation[~b_mask] = torch.sigmoid(xm_params) 

            for k_plot in range(k_plots):
                params_plots[k_plot,k] = round(imputation[missing_pattern_x[k_plot], missing_pattern_y[k_plot]],6)

            #print(params_plots[:,k].tolist())
            #test_optimizer.step(lambda: xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, p_xm = p_xm,encoder = encoder, decoder = decoder, d=d, K=1)[0]) # when using LBFGS as optim

            if binary_data:
                xhat[xhat < 0.5] = 0
                xhat[xhat >= 0.5] = 1

            ### Get mse error on imputation
            err = np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
            xm_loglikelihood.append(-loss.item())
            xm_approx_loss.append(-approx_loss.item())
            xm_mse.append(err)

            print("Iter : {}, Loss : {}".format(k+1,loss.item()))
            #print(k,err, loss)
            
            #with open(results + str(i) + "newtest.txt", 'a') as f:
            #    f.write("Loss :" + str(float(loss)) + '\n')

            loss_acc.append(loss.item())
            if (k+1)==num_epochs_test:
                with torch.no_grad():
                    img = b_full.cpu().data.numpy()         ## added .data
                    plot_image(np.squeeze(img),results + str(i) + "image1/epoch_" + str(k+1) + "xm-true.png" )
                    
                    imputation = b_data 
                    imputation[~b_mask] = torch.sigmoid(xm_params) 
                    img = imputation.cpu().data.numpy()
                    #plot_image(np.squeeze(img),results + str(i) + "image1/epoch_" +   str(k+1) + "xm-input.png", missing_pattern_x = missing_pattern_x, missing_pattern_y = missing_pattern_y )
                    plot_image(np.squeeze(img),results + str(i) + "image1/epoch_" +   str(k+1) + "xm-input.png")
                    xhat = xm_impute(iota_x = b_data,mask = b_mask,p_z=p_z, p_xm=p_xm, encoder = encoder,decoder = decoder, d=d, L=1).cpu().data.numpy().reshape(1,1,28,28)               

                    mask = b_mask.cpu().data.numpy().astype(np.bool)

                    if binary_data:
                        xhat[xhat < 0.5] = 0
                        xhat[xhat >= 0.5] = 1

                    plot_image(np.squeeze(xhat), results + str(i) + "image1/epoch_" + str(k+1) + "xm-imputed.png" )

            test_mse += float(err)
            test_log_likelihood += float(loglike)
            test_loss += float(loss)
        
        gc.collect()
        print('Test-image {} Test Loss {}, log-likelihood {}, test-mse {}'.format(nb, test_loss/k, test_log_likelihood/k, test_mse/k))
        #if (nb>=1):
        #    break

    #plot_image(np.squeeze(b_data.cpu().data.numpy()),results + str(i) + "image1/" + "pseudo-gibbs-1000.png" )

    #exit()

    end = datetime.now()
    diff = end - start

    print(cuda.current_context().get_memory_info())

    del xm_params

    print(cuda.current_context().get_memory_info())

    ### Save test log-likelihood
    #with open(results + "test.txt", 'a') as f:
    #        f.write("xm nb " + str(num_epochs_test) + "% missing  " + str(i) + "\t test_log_likelihood/img " +  str(float(test_log_likelihood/(nb))) + " \t test_log_likelihood/pixel" +  str(float(test_log_likelihood/(nb*(1-i)*28*28))) + "\t test_mse/img " + str(test_mse/(nb)) + "\t test_mse/pixel " + str(test_mse/(nb*i*28*28)) + "\t time " + str(diff.total_seconds()/60) + "\n")

    #test_log_likelihood, test_loss, test_mse = test_log_likelihood/(nb*(1-i)*28*28), test_loss/(nb), test_mse/(nb*i*28*28)

    print("memory after --" )
    print(torch.cuda.memory_allocated(device=3))

    print(torch.cuda.empty_cache())

    print("memory after deleting cache--" )
    print(torch.cuda.memory_allocated(device=3))
    #exit()

    x = np.arange(1, num_epochs_test + 1, 1)
    plt.plot( x, loss_acc)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(results+'loss-iteration.png')
    plt.close()

    #pixel_true = b_full[~b_mask].cpu().data.numpy()[check_plots]
    
    #for plot in range(k_plots):
    #    plt.plot(x, params_plots[plot,:], label=str(missing_pattern_x[plot].item()) + "," + str(missing_pattern_y[plot].item()))
    #    plt.legend(loc="upper left")
    #    plt.xlabel('Iterations')
    #    plt.ylabel('param of pixel')
    #    plt.show()
    #    plt.savefig(results+ "pixels-params/pixel" + str(plot) + '-' + str(true_pixels[plot]) + ", "  + str(missing_pattern_x[plot].item()) + "," + str(missing_pattern_y[plot].item()) + '.png')
    #    plt.close()


    x = np.arange(1, num_epochs_test + 1, 1)
    plt.plot(x, pseudo_gibbs_loss, label='Psuedo-Gibbs')
    plt.plot(x, xm_loglikelihood, color='g', label='Our method')
    plt.plot(x, m_loglikelihood, color='y', label='Metropolis-within-Gibbs')
    #plt.plot(x, xm_approx_loss, color='m', label='our method2')
    plt.axhline(y=lower_bound, color='r', linestyle='-', label='missing 0s')
    plt.axhline(y=upper_bound, color='b', linestyle='-', label='true image')
    plt.xlabel('Iterations')
    plt.ylabel('loglikelihood')
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig(results+'comparison-loglikelihood.png')
    plt.close()

    plt.plot(x, pseudo_gibbs_mse, label='Psuedo-Gibbs')
    plt.plot(x, xm_mse, color='g', label='Our method')
    plt.plot(x, m_error, color='y', label='Metropolis-within-Gibbs')
    plt.axhline(y=lower_err, color='r', linestyle='-', label='true image')
    plt.axhline(y=upper_err, color='b', linestyle='-', label='missing 0s')
    plt.xlabel('Iterations')
    plt.ylabel('mse on missing pixels')
    plt.legend(loc="upper left")
    #plt.ylim(0, 0.2)
    plt.show()
    plt.savefig(results+'comparison-mse.png')
    plt.close()
    z=0
    if z:
        ### Sample and plot one image
        for data in random.sample(list(test_loader), 1):
            b_data, b_mask, b_full = data
            b_data = b_data.to(device,dtype = torch.float)
            b_mask = b_mask.to(device,dtype = torch.float)
                    
            count +=1
            
            img = b_full.cpu().numpy()
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

