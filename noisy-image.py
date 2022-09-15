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
from inference import m_g_sampler, m_g_sampler_noise, pseudo_gibbs_noise
from networks import *
from init_methods import *
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
num_epochs_test = 100
sigma = 0.25
#results="/home/sakshia1/myresearch/missing_dasta/miwae/pytorch/results/mnist-" + str(binary_data) + "-"

##results for beta-annealing
#results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-beta-annealing-"
##Results for alpha-annealing
results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-"
ENCODER_PATH = "models/e_model_"+ str(binary_data) + ".pt"  ##without 20 is d=50
DECODER_PATH = "models/d_model_"+ str(binary_data) + ".pt"  ##simple is for simple VAE

"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""

train_loader, val_loader = train_valid_loader(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data)

"""
Initialize the network and the Adam optimizer
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

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

burn_in_period = 20

###Generate 500 samples from decoder
#for i in range(500):
#    x = generate_samples(p_z, decoder, d, L=1).cpu().data.numpy().reshape(1,1,28,28)  
#    plot_image(np.squeeze(x), os.getcwd() + "/results/generated-samples/" + str(i)+ ".png" ) 

xm_loss = np.zeros((6,num_epochs_test))
xm_loss_per_img = np.zeros((6,10,num_epochs_test))
xm_mse_per_img = np.zeros((6,10,num_epochs_test))

num_images = 10

for iterations in range(6):
    for i in [-2]:
        ### Get test loader for different missing percentage value
        #print("memory before --" )
        #print(torch.cuda.memory_allocated(device=0))
        ## MAR (0.5, 0.8)
        #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = i),batch_size=1)
        ## Right half missing (0)
        #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, top_half=True),batch_size=1)
        ## 4 patches of size 10*10 missing (-1)
        test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, noise=True, perc_miss=1, std=sigma),batch_size=1)
        print("test data loaded")
        test_log_likelihood, test_loss, test_mse, nb, = 0, 0, 0, 0
        start = datetime.now()
        #np.random.seed(5678)
        ### Optimize over test variational parameters

        ##Init xm_params with output of decoder:
        pseudo_gibbs_loss = np.zeros((num_epochs_test))
        pseudo_gibbs_mse = np.zeros((num_epochs_test))
        pseudo_gibbs_elbo = np.zeros((num_epochs_test))
        pseudo_gibbs_loglike = np.zeros((num_epochs_test))

        m_elbo = np.zeros((num_epochs_test))
        m_mse = np.zeros((num_epochs_test))
        m_loss = np.zeros((num_epochs_test))
        m_loglikelihood = np.zeros((num_epochs_test))

        xm_elbo = np.zeros((num_epochs_test))
        xm_mse = np.zeros((num_epochs_test))
        xm_loglikelihood = np.zeros((num_epochs_test))

        term1 = np.zeros((num_epochs_test))  #loglikelihood
        term2 = np.zeros((num_epochs_test))  #KL
        term3 = np.zeros((num_epochs_test))  #Entropy

        lower_bound_all = upper_bound_all = lower_log_like_all = upper_log_like_all = lower_err = upper_err = 0

        total_loss = 0

        for data in test_loader:
            nb += 1
            if nb<21:
                continue
            if nb==25:
                break
            #if nb!=22 and nb!=26:
            #    print("continuing ---", nb)
            #    continue

            print("Image : ", nb)
            b_data, b_mask, b_full = data
            #b_data_init = b_data
            b_data_start = b_data
            missing_counts = 1*28*28 - int(b_mask.sum())

            #b_data = b_data.to(device,dtype = torch.float)
            b_mask = b_mask.to(device,dtype = torch.bool)
            #b_data_init = b_data_init.to(device,dtype = torch.float)
            b_full_ = b_full.reshape([1,1,28,28]).to(device,dtype = torch.float)
            
            #b_data = torch.clamp(b_data, min=0, max=1)

            n = 0

            if n:
                ##Calculate lower and upper bound (ELBO) of likelihood for observed pixels
                lower_bound, lower_log_like =  mvae_loss(iota_x = b_data.to(device,dtype = torch.float),mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)
                upper_bound, upper_log_like =  mvae_loss(iota_x = b_full.to(device,dtype = torch.float),mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)

                print("ELBO's of image with missing 0's and true values:", -lower_bound.item(), -upper_bound.item())

                lower_bound_all += -lower_bound.item()
                upper_bound_all += -upper_bound.item()
                lower_log_like_all += lower_log_like.item()
                upper_log_like_all += upper_log_like.item()

                ##Imputation given missing input pixels with 0s
                #b_data[~b_mask] = b_data_init[~b_mask]      

                upper_imp = mvae_impute(iota_x = b_data.to(device,dtype = torch.float),mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1)[0].reshape(1,1,28,28)
                v = torch.sigmoid(upper_imp)
                imputation = b_data
                imputation[~b_mask] = v[~b_mask]
                imputation = imputation.cpu().data.numpy().reshape(28,28)
                upper_err += np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])

                ##Imputation given missing input pixels with true pixels  

                lower_imp = mvae_impute(iota_x = b_full.to(device,dtype = torch.float),mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1)[0].reshape(1,1,28,28)
                v = torch.sigmoid(lower_imp)
                imputation = b_data
                imputation[~b_mask] = v[~b_mask]
                imputation = imputation.cpu().data.numpy().reshape(28,28)
                lower_err += np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])

            #b_data = b_data_init    

            burn_in_ = True
            random = False

            if random: 
                x_logits_init = torch.zeros_like(b_data)
                p_x_m = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_init[~b_mask]),1)
                b_data_init = b_data
                b_data_init[~b_mask] = p_x_m.sample()
                x_logits_init[~b_mask] = torch.log(b_data_init/(1-b_data_init))[~b_mask]
            elif burn_in_:
                x_logits_init, b_data_init = burn_in_noise(b_data_start.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, burn_in_period=burn_in_period, sigma=sigma, nb=nb)
                x_logits_init = x_logits_init.cpu()
                b_data_init = b_data_init.cpu()
                #x_logits_init, b_data_init = burn_in(b_data, b_mask, encoder, decoder, p_z, d, burn_in_period=burn_in_period)
            else:
                b_data_init = b_data
                b_data_init[~b_mask] = k_neighbors_sample(b_data, b_mask) 
                b_data_init = b_data_init
                x_logits_init = torch.log(b_data_init/(1-b_data_init))

            plot_image(np.squeeze(torch.sigmoid(x_logits_init).reshape(28,28).cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + "after-burn-in.png" )

            #plot_image(np.squeeze(b_data_init.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "k-neighbors.png" )
            #b_data[~b_mask] = torch.sigmoid(x_logits_burn[~b_mask])

            #err += np.array([mse(b_data.cpu().data.numpy(),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy())])
            #print("Burn-in imputed m.s.e")

            ### Sample an image from burn-in logits and use it for all inference algorithms
            sampled_image = b_data_init
            #sampled_image[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_init),1).sample()[~b_mask]
            sampled_image_m = sampled_image
            sampled_image_o = sampled_image


            a = b_full
            if iterations==0:
                #sampled_image[~b_mask] = b_full[~b_mask]    
                sampled_image = a
                sampled_image_m = sampled_image
                x_logits_init = torch.log(a/(1-a))
            else:
                sampled_image = b_data_init
                sampled_image_m = sampled_image
                sampled_image_o = sampled_image


            #b_data = sampled_image
            if not burn_in_ : #or iterations==0
                plot_image(np.squeeze(sampled_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "init.png" )
            else:
                #burn_in_image = b_data_init
                burn_in_image = torch.sigmoid(x_logits_init)
                #burn_in_image = torch.sigmoid(x_logits_init)
                plot_image(np.squeeze(burn_in_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "init.png" )

            # Pseudo-gibbs sampler 
            x_prev = sampled_image
            x_logits_prev = x_logits_init
            x_logits_prev_m = x_logits_init
            y_prev = b_data

            
            x_logits_pseudo_gibbs, x_sample_pseudo_gibbs = pseudo_gibbs_noise(x_prev.to(device,dtype = torch.float), x_logits_prev.to(device,dtype = torch.float), b_data, b_mask, encoder, decoder, p_z, d, T=num_epochs_test, sigma=sigma, nb=nb)

            plot_image(np.squeeze(torch.sigmoid(x_logits_init).reshape(28,28).cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + "after-pseudo-gibbs.png" )
            ##suspicious change of true image here.. 
            #b_full = b_full_

            #Impute image with pseudo-gibbs
            #pseudo_gibbs_image = b_data
            pseudo_gibbs_image = torch.sigmoid(x_logits_pseudo_gibbs)
            #pseudo_gibbs_image[~b_mask] = torch.sigmoid(x_logits_pseudo_gibbs[~b_mask])
            plot_image(np.squeeze(pseudo_gibbs_image.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + "pseudo-gibbs.png" )

            ##M-with-gibbs sampler
            m_nelbo, m_error, x_full_logits, m_loglike =  m_g_sampler_noise(iota_x = b_data.to(device,dtype = torch.float), missing = sampled_image_m.to(device,dtype = torch.float), all_logits = x_logits_prev_m.to(device,dtype = torch.float), full = b_full.to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, sigma =sigma, d=d, K=1, T=num_epochs_test)

            m_loss += np.array(m_nelbo).reshape((num_epochs_test))
            m_loglikelihood += np.array(m_loglike).reshape((num_epochs_test))
            m_elbo -= np.array(m_nelbo).reshape((num_epochs_test))
            m_mse += np.array(m_error).reshape((num_epochs_test))

            #Impute image with metropolis-within-pseudo-gibbs
            #metropolis_image = b_data
            metropolis_image = torch.sigmoid(x_full_logits)
            #metropolis_image[~b_mask] = torch.sigmoid(x_full_logits[~b_mask])
            plot_image(np.squeeze(metropolis_image.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + "metropolis-within-pseudo-gibbs.png" )

            ##Our method
            a = b_full

            if iterations==0:
                xm_params = torch.log(a/(1-a)).reshape(-1)
            else:
                xm_params = x_logits_init.reshape(-1)

            plot_image(np.squeeze(torch.sigmoid(xm_params).reshape(28,28).cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + "xm_params_init.png" )

            #print(xm_params)
            #exit()
            print("Image: ", nb, " iter: ", iterations, xm_params)

            xm_params = xm_params.to(device)
            xm_params = torch.clamp(xm_params, min=-10, max=10)
            p_xm = td.Independent(td.ContinuousBernoulli(logits=(xm_params).cuda()),1)       ##changed to probs, since sigmoid outputs  
            
            xm_params.requires_grad = True
            #test_optimizer = torch.optim.LBFGS([xm_params]) ##change, sgd, adagrad
            test_optimizer = torch.optim.Adam([xm_params], lr=1.0, betas=(0.9, 0.999)) 
            #test_optimizer = torch.optim.Adagrad([xm_params], lr=0.1) 
            
            #b_data = b_data_init
            beta_0 = 1

            p = 0

            if p:
                k_plots= 50
                missing_pattern = np.argwhere(b_mask.reshape(-1).cpu().data.numpy() == False)
                indices_1 = np.argwhere((b_full.reshape(-1).cpu().data.numpy()[missing_pattern] > 0.5))

                if k_plots<len(indices_1):     
                    check_plots = np.random.choice(len(indices_1), k_plots, replace=False)
                else:
                    check_plots = np.arange(0, len(indices_1), 1, dtype=int)
                check_plots = np.asarray(check_plots).astype(np.int32)
                k_plots = len(check_plots)
            
                true_pixels = b_full.cpu().data.numpy().reshape(-1)[missing_pattern][indices_1]
                params_plots = np.zeros((k_plots, num_epochs_test))

            #b_full = b_full_3

            plot_image(np.squeeze(b_full.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "after-loop.png" )

            #exit() 


            for k in range(num_epochs_test):
                #x_logits = xm_params.detach()
                print(xm_params)
                #b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample().reshape(-1)
                #print(b_data[~b_mask])
                #exit()
                beta = beta_0

                #beta = beta/2 + 0.5
                #beta = max(beta_0 - (9/49)*k,1)
                #beta = max(beta/4,1)
                #beta = max(beta/1.7, 1)
                #beta = min(beta_0 + (0.999/49)*k ,1)

                test_optimizer.zero_grad()
                #loss, log_like, aa, bb, cc = xm_loss_(iota_x = b_data, sampled = sampled_image_o, mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta = beta, K=1, K_z= 1, epoch=k)
                loss, log_like, aa, bb, cc = xm_loss_noise(iota_x = b_data.to(device,dtype = torch.float), mask = b_mask, p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta=beta,K=1, K_z= 1, std=sigma)

                loss.backward()            
                test_optimizer.step()

                ##Calculate loglikelihood and mse on imputation
                imputation = b_data
                v = torch.sigmoid(xm_params).detach()
                imputation = v.reshape(*b_data.shape)
                #loss, log_like, aa, bb, cc = xm_loss_(iota_x = imputation, sampled = imputation, mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta = beta, K=1, K_z= 1, train=False, epoch=k)
                loss, log_like, aa, bb, cc = xm_loss_noise(iota_x = imputation.to(device,dtype = torch.float),mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta=beta, K=1, K_z= 1, std=sigma)

                xm_loglikelihood[k] += log_like.item()

                #xm_params_updated = xm_params.detach()
                #p_xm = td.continuous_bernoulli.ContinuousBernoulli(logits=xm_params_updated.reshape([-1,1]))
                #sampled_image_o[~b_mask] = p_xm.rsample([1]).reshape(-1)   

                ### Get mse error on imputation
                imputation = imputation.cpu().data.numpy().reshape(28,28)
                err = np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])
                xm_mse[k] += err
                #loss, log_like, aa, bb, cc = xm_loss_(iota_x = b_data, sampled = sampled_image_o, mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta = beta, K=100, K_z= 1, epoch=k)
                loss, log_like, aa, bb, cc = xm_loss_noise(iota_x = b_data.to(device,dtype = torch.float),mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta=beta, K=100, K_z= 1, std=sigma)

                term1 += aa.item()
                term2 += bb.item()
                term3 += cc.item()

                xm_loss[iterations,k] += loss.item()
                xm_elbo[k] += -loss.item()
                xm_loss_per_img[iterations, nb%10, k] = loss.item() 
                xm_mse_per_img[iterations, nb%10, k] = err

                if(k==num_epochs_test-1):
                    #print(log_like.item())
                    print("Loss with our method: ", loss.item())

                if p:
                    for k_plot in check_plots:
                    #    print(check_plots)
                        params_plots[k_plot,k] = round(imputation[missing_pattern_x[k_plot], missing_pattern_y[k_plot]],6)
                        #params_plots[k_plot,k] = round(xm_params[indices_1[k_plot]].item(),6)
                    #print(params_plots[:,k].tolist())
                #test_optimizer.step(lambda: xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, p_xm = p_xm,encoder = encoder, decoder = decoder, d=d, K=1)[0]) # when using LBFGS as optim
                
                if binary_data:
                    xhat[xhat < 0.5] = 0
                    xhat[xhat >= 0.5] = 1

                if (k+1)==num_epochs_test or (k+1)%10==0 or k<10:
                    with torch.no_grad():
                        img = b_full.cpu().data.numpy()         ## added .data
                        plot_image(np.squeeze(img),results + str(i) + "/images/" + str(nb%10) + "/"  +  "xm-true.png" )
                        missing = b_data_start
                        #missing[~b_mask] = 0.5      
                        img = missing.cpu().data.numpy() 
                        plot_image(np.squeeze(img),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) +  "missing.png" )

                        imputation = b_data 
                        imputation = torch.sigmoid(xm_params).reshape(*b_data.shape) 
                        img = imputation.cpu().data.numpy()

                        #plot_image(np.squeeze(img),results + str(i) + "image1/epoch_" +   str(k+1) + "xm-input.png", missing_pattern_x = missing_pattern_x, missing_pattern_y = missing_pattern_y )
                        
                        plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "xm-input.png")
                        
                #test_log_likelihood += float(loglike)
                #test_loss += float(loss)
            
            x = np.arange(1, num_epochs_test + 1, 1).reshape(num_epochs_test)  
            plt.plot(x, xm_loss_per_img[iterations, nb%10])
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            #plt.ylim(-1500, -1300)
            plt.show()
            plt.savefig(results + str(i) + "/loss/image-" + str(nb%10) + 'loss-iteration.png')
            plt.close()

            image0 = plt.imread(results + str(i) + "/loss/image-" + str(nb%10) + 'loss-iteration.png')

            image1 = plt.imread(results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "missing.png" )
            image2 = plt.imread(results + str(i) + "/images/" + str(nb%10) + "/"  +  "xm-true.png" )
            image3 = plt.imread(results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "init.png" )

            image4 = plt.imread(results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + "pseudo-gibbs.png" )
            image5 = plt.imread(results + str(i) + "/images/" + str(nb%10) + "/" +  str(iterations) + "metropolis-within-pseudo-gibbs.png" )
            image6 = plt.imread(results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(k) + "xm-input.png")

            image7 = plt.imread(results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(9) + "xm-input.png")
            image8 = plt.imread(results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' + str(49) + "xm-input.png")        

            plot_images(image1,image2, image3, image4, image5, image6, image7, image8, image0, results + str(i) + "/compiled/image" + str(nb%10) + str(iterations) +".pdf")
            gc.collect()
            #print('Test-image {}, log-likelihood {}, test-mse {}'.format(nb, xm_loglikelihood, xm_mse))
            
            x = np.arange(1, num_epochs_test + 1, 1).reshape(num_epochs_test)
            #for plot in range(k_plots):
                    #plt.plot(x, params_plots[plot,:], label=str(missing_pattern_x[plot].item()) + "," + str(missing_pattern_y[plot].item()))
            #        plt.plot(x, params_plots[plot,:], label=str(check_plots[plot].item()) )
            #        plt.legend(loc="upper left")
            #        plt.xlabel('Iterations')
            #        plt.ylabel('pixel logits')
            #        plt.show()
            #        plt.savefig(results + str(i) + "/images/" + str(nb%10) +  "/pixel" + str(plot) + '-' + str(true_pixels[plot]) + ", "  + str(missing_pattern[plot].item()) + '.png')
            #        plt.close()
            #break

            #if nb==26:
            #    break

            if (nb==29):
                end = datetime.now()
                diff = end - start
                del xm_params
                
                #loss_acc /= nb
                #plt.plot(x, loss_acc)
                #plt.xlabel('Iterations')
                #plt.ylabel('Loss')
                #plt.ylim(-200, 0)
                #plt.show()
                #plt.savefig(results + str(i) + "/loss/" + 'loss-iteration.png')
                #plt.close()
                #pixel_true = b_full[~b_mask].cpu().data.numpy()[check_plots]
                #loss_acc /= nb

                pseudo_gibbs_loss /= num_images
                xm_loglikelihood /= num_images
                lower_bound_all /= num_images
                upper_bound_all /= num_images
                pseudo_gibbs_mse /= num_images
                xm_mse /= num_images
                m_mse /= num_images
                lower_err /= num_images
                upper_err /= num_images
                xm_loss[iterations] /= num_images
                xm_elbo /= num_images
                pseudo_gibbs_elbo /= num_images
                m_elbo /= num_images
                m_loss /= num_images
                pseudo_gibbs_loglike /= num_images
                m_loglikelihood /= num_images
                upper_log_like_all /= num_images
                lower_log_like_all /= num_images

                term1= term1/num_images
                term2 = term2/num_images
                term3 = term3/num_images

                print("Average loss: ")
                print(xm_loss[iterations,-1])
                #print(xm_mse)
                #x = np.arange(1, num_epochs_test + 1, 1)
                """
                plt.plot(x, pseudo_gibbs_loss, label='Psuedo-Gibbs')
                plt.plot(x, xm_loglikelihood, color='g', label='Our method')
                plt.plot(x, m_loglikelihood_all, color='y', label='Metropolis-within-Gibbs')
                #plt.plot(x, xm_approx_loss, color='m', label='our method2')
                plt.axhline(y=lower_bound_all, color='r', linestyle='-', label='missing 0s')
                plt.axhline(y=upper_bound_all, color='b', linestyle='-', label='true image')
                plt.xlabel('Iterations')
                plt.ylabel('loglikelihood')
                plt.legend(loc="upper left")
                plt.show()
                plt.savefig(results+'comparison-loglikelihood-try.png')
                plt.close()
                """

                ## Plot ELBOs
    #            plt.plot(x, pseudo_gibbs_elbo, label='Psuedo-Gibbs')
    #            plt.plot(x, xm_elbo, color='g', label='Our method')
    #            plt.plot(x, m_elbo, color='y', label='Metropolis-within-Gibbs')
    #            plt.axhline(y=lower_bound_all, color='r', linestyle='-', label='missing 0s')
    #            plt.axhline(y=upper_bound_all, color='b', linestyle='-', label='true image')
    #            plt.xlabel('Iterations')
    #            plt.ylabel('ELBO') 
    #            plt.legend(loc="lower left")
    #            plt.show()
    #            plt.savefig(results + str(i) + "/" + 'ELBO.png')
    #            plt.close()


                
                ##Plot Joint log-likelihood
                plt.plot(x, pseudo_gibbs_loglike, label='Psuedo-Gibbs')
                plt.plot(x, xm_loglikelihood, color='g', label='Our method')
                #plt.plot(x, xm_loglikelihood, color='m', label='Our method - ELBO')
                plt.plot(x, m_loglikelihood, color='y', label='Metropolis-within-Gibbs')
                #plt.plot(x, xm_approx_loss, color='m', label='our method2')
                plt.axhline(y=lower_log_like_all, color='r', linestyle='-', label='missing 0s')
                plt.axhline(y=upper_log_like_all, color='b', linestyle='-', label='true image')
                #plt.ylim(lower_log_like_all - 100, 2000)
                plt.xlabel('Iterations')
                plt.ylabel('joint loglikelihood') 
                plt.legend(loc="lower left")
                plt.show()
                plt.savefig(results + str(i) + "/" + str(iterations) + 'comparison-loglikelihood.png')
                plt.close()

                plt.plot(x, pseudo_gibbs_mse, label='Psuedo-Gibbs')
                plt.plot(x, xm_mse, color='g', label='Our method')
                plt.plot(x, m_mse, color='y', label='Metropolis-within-Gibbs')
                plt.axhline(y=lower_err, color='r', linestyle='-', label='true image')
                plt.axhline(y=upper_err, color='b', linestyle='-', label='missing 0s')
                plt.xlabel('Iterations')
                plt.ylabel('avg mse per pixel')
                plt.legend(loc="lower left")
                #plt.ylim(0, 0.2)
                plt.show()
                plt.savefig(results + str(i) + "/" + str(iterations) + 'comparison-mse.png')
                plt.close()

                plt.plot(x, term1)
                plt.xlabel('Iterations')
                plt.ylabel('term1')
                plt.show()
                plt.savefig(results + str(i) + "/loss/" +  'loss-term1.png')
                plt.close()

                plt.plot(x, term2)
                plt.xlabel('Iterations')
                plt.ylabel('term2')
                plt.show()
                plt.savefig(results + str(i) + "/loss/" + 'loss-term2.png')
                plt.close()

                plt.plot(x, term3)
                plt.xlabel('Iterations')
                plt.ylabel('term3')
                plt.show()
                plt.savefig(results + str(i) + "/loss/" +'loss-term3.png')
                plt.close()

                pseudo_gibbs_loss *= num_images
                xm_loglikelihood *= num_images
                lower_bound_all *= num_images
                upper_bound_all *= num_images
                pseudo_gibbs_mse *= num_images
                xm_mse *= num_images
                m_mse *= num_images
                lower_err *= num_images
                upper_err *= num_images
                xm_loss[iterations] *= num_images
                xm_elbo *= num_images
                pseudo_gibbs_elbo *= num_images
                m_elbo *= num_images
                m_loss *= num_images
                pseudo_gibbs_loglike *= num_images
                m_loglikelihood *= num_images
                upper_log_like_all *= num_images
                lower_log_like_all *= num_images

                term1= term1*num_images
                term2 = term2*num_images
                term3 = term3*num_images

            
                #x = np.arange(1, num_epochs_test + 1, 1).reshape(num_epochs_test)  
                #plt.plot(x, loss_acc)
                #plt.xlabel('Iterations')
                #plt.ylabel('Loss')
                #plt.ylim(-200, 0)
                #plt.show()
                #plt.savefig(results + str(i) + "/loss/"  +'loss-iteration.png')
                #plt.close()

                
                #loss_acc *= nb

                #image1 = plt.imread(results + str(i) + "/loss/"  +'loss-iteration.png')
                #image2 = plt.imread(results + str(i) + "/" +'comparison-mse.png')
                #image3 = plt.imread(results + str(i) + "/" + 'comparison-loglikelihood.png')

                #plot_all_averages(image2, image3, results + str(i) + "/averages.png")
                break

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


x = np.arange(1, num_epochs_test + 1, 1).reshape(num_epochs_test)

colours = ['g', 'b', 'y', 'r', 'k', 'c']
for k_iters in range(6):
    #print(xm_loss[k_iters],colours[k_iters])
    if k_iters==0:
        plt.plot(x, xm_loss[k_iters]/num_images, color=colours[k_iters], label="true image")
    else:
        plt.plot(x, xm_loss[k_iters]/num_images, color=colours[k_iters], label="our method " + str(k_iters))
plt.xlabel('Iterations')
plt.ylabel('Loss (NELBO)') 
#plt.ylim(250, 1000)
plt.legend(loc="upper left")
plt.show()
plt.savefig(results + str(i) + "/compiled/" + 'loss.png')
plt.close()


for image in range(num_images):
    for k_iters in range(6):
        #print(xm_loss_per_img[k_iters, image],colours[k_iters])
        if k_iters==0:
            plt.plot(x, xm_loss_per_img[k_iters, image], color=colours[k_iters], label="true image")
        else:
            plt.plot(x, xm_loss_per_img[k_iters, image], color=colours[k_iters], label="our method " + str(k_iters))
    plt.xlabel('Iterations')
    plt.ylabel('Loss (NELBO)') 
    plt.ylim(100, 1000)
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig(results + str(i) + "/compiled/" + str(image)  + '-loss.png')
    plt.close()

for image in range(num_images):
    for k_iters in range(6):
        #print(xm_mse_per_img[k_iters, image],colours[k_iters])
        if k_iters==0:
            plt.plot(x, xm_mse_per_img[k_iters, image], color=colours[k_iters], label="true image")
        else:
            plt.plot(x, xm_mse_per_img[k_iters, image], color=colours[k_iters], label="our method " + str(k_iters))
    plt.xlabel('Iterations')
    plt.ylabel('MSE') 
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig(results + str(i) + "/compiled/" + str(image)  + '-mse.png')
    plt.close()


end = datetime.now()
diff = end - start

print("time taken --")
print(diff)