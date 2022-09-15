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
#results="/home/sakshia1/myresearch/missing_dasta/miwae/pytorch/results/mnist-" + str(binary_data) + "-"

results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-"
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
for i in [-1]:
    ### Get test loader for different missing percentage value
    #print("memory before --" )
    #print(torch.cuda.memory_allocated(device=0))
    ## MAR (0.5, 0.8)
    #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = i),batch_size=1)
    ## Right half missing (0)
    #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, top_half=True),batch_size=1)
    ## 4 patches of size 10*10 missing (-1)
    test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, patches=True),batch_size=1)
    
    print("test data loaded")
    test_log_likelihood, test_loss, test_mse, nb, = 0, 0, 0, 0
    start = datetime.now()
    np.random.seed(5678)
    ### Optimize over test variational parameters

    ##Init xm_params with output of decoder:
    pseudo_gibbs_loss = np.zeros((num_epochs_test))
    pseudo_gibbs_mse = np.zeros((num_epochs_test))
    pseudo_gibbs_loglike = np.zeros((num_epochs_test))

    m_loglikelihood_all = np.zeros((num_epochs_test))
    m_error_all = np.zeros((num_epochs_test))
    m_nelbo_all = np.zeros((num_epochs_test))

    loss_acc = np.zeros((num_epochs_test))
    xm_loglikelihood = np.zeros((num_epochs_test))
    xm_mse = np.zeros((num_epochs_test))
    xm_approx_loss = np.zeros((num_epochs_test))

    lower_bound_all = upper_bound_all = lower_log_like_all = upper_log_like_all = lower_err = upper_err = 0

    err = 0
    for data in test_loader:
        nb += 1
        print("Image : ", nb)
        b_data, b_mask, b_full = data
        b_data_init = b_data
        missing_counts = 1*28*28 - int(b_mask.sum())
        #b_full = b_full.reshape(28,28)
        #print(b_data,b_mask)
        b_data = b_data.to(device,dtype = torch.float)
        b_mask = b_mask.to(device,dtype = torch.bool)
        b_data_init = b_data_init.to(device,dtype = torch.float)
        #b_data[~b_mask] = 0
        b_full = b_full.reshape([1,1,28,28]).to(device,dtype = torch.float)

        ##Calculate lower and upper bound of likelihood for observed pixels
        lower_bound, lower_log_like =  mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)
        upper_bound, upper_log_like =  mvae_loss(iota_x = b_full,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)

        lower_bound_all += -lower_bound.item()
        upper_bound_all += -upper_bound.item()
        lower_log_like_all += lower_log_like.item()
        upper_log_like_all += upper_log_like.item()

        #print(lower_log_like_all, upper_log_like_all)
        ##Imputation given missing input pixels with 0s
        b_data[~b_mask] = b_data_init[~b_mask]      
        upper_imp = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)
        v = torch.sigmoid(upper_imp)
        imputation = b_data
        imputation[~b_mask] = v[~b_mask]
        imputation = imputation.cpu().data.numpy().reshape(28,28)
        upper_err += np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])

        ##Imputation given missing input pixels with true pixels       
        lower_imp = mvae_impute(iota_x = b_full,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)
        v = torch.sigmoid(lower_imp)
        imputation = b_data
        imputation[~b_mask] = v[~b_mask]
        imputation = imputation.cpu().data.numpy().reshape(28,28)
        lower_err += np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])

        b_data[~b_mask] = b_data_init[~b_mask]      

        # Burning in samples
        for l in range(burn_in_period):
            #print(l)
            x_logits = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)
            b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
            #torch.sigmoid(x_logits[~b_mask])
        x_logits_burn = x_logits

        b_data[~b_mask] = torch.sigmoid(x_logits_burn[~b_mask])
        plot_image(np.squeeze(b_data.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + "burn-in.png" )
        err += np.array([mse(b_data.cpu().data.numpy(),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy())])
        #print("Burn-in imputed m.s.e")

        # Pseudo-gibbs sampler 
        for l in range(num_epochs_test):
            ## To perform Gibbs sampling
            b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
            loss, loglike = mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)
            x_logits = mvae_impute(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z = p_z, d=d, L=1).reshape(1,1,28,28)
            pseudo_gibbs_loss[l] += -loss.item()

            ## To calculate likelihood & error on imputed image
            b_data[~b_mask] = torch.sigmoid(x_logits[~b_mask])
            loss, loglike = mvae_loss(iota_x = b_data,mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1)
            pseudo_gibbs_loglike[l] += loglike.item()
            err = np.array([mse(b_data.cpu().data.numpy(),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy())])
            pseudo_gibbs_mse[l] += err  

        #print(pseudo_gibbs_loglike)
        #Impute image with pseudo-gibbs
        pseudo_gibbs_image = b_data
        pseudo_gibbs_image[~b_mask] = torch.sigmoid(x_logits[~b_mask])
        plot_image(np.squeeze(pseudo_gibbs_image.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + "pseudo-gibbs.png" )

        ##M-with-gibbs sampler
        x_logits = x_logits_burn
        b_data[~b_mask] = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits),1).sample()[~b_mask]
        m_nelbo, m_error, x_full_logits, m_loglikelihood =  m_g_sampler(iota_x = b_data, full = b_full, mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=1, T=num_epochs_test)

        #print(m_loglikelihood)
        m_nelbo_all += np.array(m_nelbo).reshape((num_epochs_test))
        m_loglikelihood_all += np.array(m_loglikelihood).reshape((num_epochs_test))
        m_error_all += np.array(m_error).reshape((num_epochs_test))

        #print(m_loglikelihood_all)
        #Impute image with metropolis-within-pseudo-gibbs
        metropolis_image = b_data
        metropolis_image[~b_mask] = torch.sigmoid(x_full_logits[~b_mask])
        plot_image(np.squeeze(metropolis_image.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" +  "metropolis-within-pseudo-gibbs.png" )

        ## Clamp with lower prob=0.05, max prob = 0.95
        #xm_params = torch.clamp(xm_params, min=-2.94, max=2.94)
        #min p = 0.01

        xm_params = x_logits_burn[~b_mask] 
        #print(xm_params)
        #xm_params = torch.log(b_full[~b_mask]/(torch.ones(missing_counts).reshape(-1)-b_full[~b_mask]))
        xm_params = torch.clamp(xm_params, min=-10, max=10)
        #xm_params = 40*torch.rand(missing_counts) - 20
        xm_params = xm_params.to(device)

        #print(xm_params.tolist())
        #upper_prob = torch.tensor(1, dtype=torch.float64) - torch.tensor(float(1.8)*np.exp(-70), dtype=torch.float64)
        #lower_prob = torch.tensor(float(1.8)*np.exp(-70), dtype=torch.float64)
        #print(upper_prob, lower_prob)
        #print("***********************")

        p_xm = td.Independent(td.ContinuousBernoulli(logits=(xm_params).cuda()),1)       ##changed to probs, since sigmoid outputs     
        #p_xm = td.Independent(td.ContinuousBernoulli(probs=(upper_prob*torch.sigmoid(xm_params)+lower_prob).cuda()),1) 
        #image = b_data
        #a = p_xm.rsample()
        #print(a.tolist(),a.shape)
        #image[~b_mask] = a.reshape(-1)   
        #print(image[~b_mask].tolist())
        #plot_image(np.squeeze(image.cpu().data.numpy()), results + str(i) + "image1/epoch_0" + "xm-initial-sampled.png" )

        xm_params.requires_grad = True
        #test_optimizer = torch.optim.LBFGS([xm_params]) ##change, sgd, adagrad
        test_optimizer = torch.optim.Adam([xm_params], lr=0.1, betas=(0.9, 0.999)) 
        #test_optimizer = torch.optim.Adagrad([xm_params], lr=0.1) 
        #xm_params_constrained = torch.clamp(xm_params, min=-100, max=100)
        #xm_params_constrained.requires_grad = False

        term1 = []
        term2 = []
        term3 = [] 
        term4 = []
        term5 = []

        beta = 1

        for k in range(num_epochs_test):
            print(k)
            print(xm_params)
            #beta -= (k)*(9/199)
            test_optimizer.zero_grad()
            loss, log_like, aa, bb, cc, dd, ee = xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta = beta, K=1, K_z= 1)
            loss.backward()            
            test_optimizer.step()

            imputation = b_data
            v = torch.sigmoid(xm_params).detach()
            imputation[~b_mask] = v

            loss, log_like, aa, bb, cc, dd, ee = xm_loss(iota_x = imputation,mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta = beta, K=1, K_z= 1, train=False)
            xm_approx_loss[k] += log_like.item()
            imputation = imputation.cpu().data.numpy().reshape(28,28)

            loss, approx_loss, aa, bb, cc, dd, ee = xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, xm_params = xm_params, encoder = encoder, decoder = decoder , device= device, d=d, beta = beta, K=100, K_z= 1)
            term1.append(aa.item())
            term2.append(bb.item())
            term3.append(cc.item())
            term4.append(dd.item())
            term5.append(ee.item())

            #imputation[~b_mask] = torch.sigmoid(xm_params) 
            #for k_plot in range(k_plots):
            #    params_plots[k_plot,k] = round(imputation[missing_pattern_x[k_plot], missing_pattern_y[k_plot]],6)
            #print(params_plots[:,k].tolist())
            #test_optimizer.step(lambda: xm_loss(iota_x = b_data,mask = b_mask,p_z = p_z, p_xm = p_xm,encoder = encoder, decoder = decoder, d=d, K=1)[0]) # when using LBFGS as optim
            if binary_data:
                xhat[xhat < 0.5] = 0
                xhat[xhat >= 0.5] = 1
            ### Get mse error on imputation
            err = np.array([mse(imputation.reshape([1,1,28,28]),b_full.cpu().data.numpy(),b_mask.cpu().data.numpy().astype(bool))])

            xm_loglikelihood[k] += -loss.item()
            xm_mse[k] += err
            #print("Iter : {}, Loss : {}".format(k+1,loss.item()))
            #print(k,err, loss)
            #with open(results + str(i) + "newtest.txt", 'a') as f:
            #    f.write("Loss :" + str(float(loss)) + '\n')
            loss_acc[k] += loss.item()
            #print(loss_acc)
            if (k+1)==num_epochs_test:
                with torch.no_grad():
                    img = b_full.cpu().data.numpy()         ## added .data
                    plot_image(np.squeeze(img),results + str(i) + "/images/" + str(nb%10) + "/"  + "xm-true.png" )
                    b_data[~b_mask] = b_data_init[~b_mask]      
                    img = b_data.cpu().data.numpy() 
                    plot_image(np.squeeze(img),results + str(i) + "/images/" + str(nb%10) + "/"  + "missing.png" )
                    imputation = b_data 
                    imputation[~b_mask] = torch.sigmoid(xm_params) 
                    img = imputation.cpu().data.numpy()
                    #plot_image(np.squeeze(img),results + str(i) + "image1/epoch_" +   str(k+1) + "xm-input.png", missing_pattern_x = missing_pattern_x, missing_pattern_y = missing_pattern_y )
                    plot_image(np.squeeze(img),results + str(i) + "/images/" +  str(nb%10) + "/"  + "xm-input.png")
                    xhat = xm_impute(iota_x = b_data,mask = b_mask,p_z=p_z, p_xm=p_xm, encoder = encoder,decoder = decoder, d=d, L=1).cpu().data.numpy().reshape(1,1,28,28)               
                    mask = b_mask.cpu().data.numpy().astype(np.bool)

                    if binary_data:
                        xhat[xhat < 0.5] = 0
                        xhat[xhat >= 0.5] = 1
                    plot_image(np.squeeze(xhat), results + str(i) + "/images/" + str(nb%10) + "/xm-imputed.png" )
            #test_mse += float(err)
            test_log_likelihood += float(loglike)
            test_loss += float(loss)
        
        #print(xm_approx_loss)
        gc.collect()
        #print('Test-image {}, log-likelihood {}, test-mse {}'.format(nb, xm_loglikelihood, xm_mse))
        
        if (nb==10):
            #plot_image(np.squeeze(b_data.cpu().data.numpy()),results + str(i) + "image1/" + "pseudo-gibbs-1000.png" )
            #exit()
            end = datetime.now()
            diff = end - start
            #print(cuda.current_context().get_memory_info())
            del xm_params
            #print(cuda.current_context().get_memory_info())
            ### Save test log-likelihood
            #with open(results + "test.txt", 'a') as f:
            #      f.write("xm nb " + str(num_epochs_test) + "% missing  " + str(i) + "\t test_log_likelihood/img " +  str(float(test_log_likelihood/(nb))) + " \t test_log_likelihood/pixel" +  str(float(test_log_likelihood/(nb*(1-i)*28*28))) + "\t test_mse/img " + str(test_mse/(nb)) + "\t test_mse/pixel " + str(test_mse/(nb*i*28*28)) + "\t time " + str(diff.total_seconds()/60) + "\n")
            #test_log_likelihood, test_loss, test_mse = test_log_likelihood/(nb*(1-i)*28*28), test_loss/(nb), test_mse/(nb*i*28*28)
            #print("memory after --" )
            #print(torch.cuda.memory_allocated(device=3))
            #print(torch.cuda.empty_cache())
            #print("memory after deleting cache--" )
            #print(torch.cuda.memory_allocated(device=3))
            loss_acc /= nb
            #print("Printing after ")
            #print(loss_acc)
            x = np.arange(1, num_epochs_test + 1, 1).reshape(num_epochs_test)
            
            plt.plot(x, loss_acc)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.show()
            plt.savefig(results + str(i) + "/loss/" + 'loss-iteration.png')
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

            loss_acc /= nb
            pseudo_gibbs_loss /= nb
            xm_loglikelihood /= nb
            lower_bound /= nb
            upper_bound /= nb
            upper_bound_all /= nb
            pseudo_gibbs_mse /= nb
            xm_mse /= nb
            m_error_all /= nb
            lower_err /= nb
            upper_err /= nb
            m_nelbo_all /= nb

            xm_approx_loss /= nb
            pseudo_gibbs_loglike /= nb
            m_loglikelihood_all /= nb
            upper_log_like_all /= nb
            lower_log_like_all /= nb
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
            plt.plot(x, pseudo_gibbs_loglike, label='Psuedo-Gibbs')
            plt.plot(x, xm_approx_loss, color='g', label='Our method')
            #plt.plot(x, xm_loglikelihood, color='m', label='Our method - ELBO')
            plt.plot(x, m_loglikelihood_all, color='y', label='Metropolis-within-Gibbs')
            #plt.plot(x, xm_approx_loss, color='m', label='our method2')
            plt.axhline(y=lower_log_like_all, color='r', linestyle='-', label='missing 0s')
            plt.axhline(y=upper_log_like_all, color='b', linestyle='-', label='true image')
            plt.xlabel('Iterations')
            plt.ylabel('joint loglikelihood') 
            plt.legend(loc="lower left")
            plt.show()
            plt.savefig(results + str(i) + "/" + 'comparison-loglikelihood.png')
            plt.close()

            plt.plot(x, pseudo_gibbs_mse, label='Psuedo-Gibbs')
            plt.plot(x, xm_mse, color='g', label='Our method')
            plt.plot(x, m_error_all, color='y', label='Metropolis-within-Gibbs')
            plt.axhline(y=lower_err, color='r', linestyle='-', label='true image')
            plt.axhline(y=upper_err, color='b', linestyle='-', label='missing 0s')
            plt.xlabel('Iterations')
            plt.ylabel('avg mse per pixel')
            plt.legend(loc="lower left")
            #plt.ylim(0, 0.2)
            plt.show()
            plt.savefig(results + str(i) + "/" +'comparison-mse.png')
            plt.close()

            loss_acc *= nb
            pseudo_gibbs_loss *= nb
            xm_loglikelihood *= nb
            m_loglikelihood_all *= nb
            lower_bound *= nb
            upper_bound *= nb
            pseudo_gibbs_mse *= nb
            xm_mse *= nb
            m_error_all *= nb
            lower_err *= nb
            upper_err *= nb

            m_nelbo_all *= nb

            xm_approx_loss *= nb
            pseudo_gibbs_loglike *= nb
            upper_log_like_all *= nb
            lower_log_like_all *= nb

            loss_acc /= nb
            #print("Printing after ")
            #print(loss_acc)
            x = np.arange(1, num_epochs_test + 1, 1).reshape(num_epochs_test)  
            plt.plot(x, loss_acc)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.show()
            plt.savefig(results + str(i) + "/loss/"  +'loss-iteration.png')
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
            loss_acc *= nb

            #plt.plot(x, term4)
            #plt.xlabel('Iterations')
            #plt.ylabel('term4')
            #plt.show()
            #plt.savefig(results + str(i) + "/loss/" +'loss-term4.png')
            #plt.close()

            #plt.plot(x, term5)
            #plt.xlabel('Iterations')
            #plt.ylabel('term5')
            #plt.show()
            #plt.savefig(results + str(i) + "/loss/" +'loss-term5.png')
            #plt.close()
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



end = datetime.now()
diff = end - start

print("time taken --")
print(diff)