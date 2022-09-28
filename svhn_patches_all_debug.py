cuda_n=1
import os
from numba import cuda
cuda.select_device(cuda_n)
print(cuda.current_context().get_memory_info())
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["NVIDIA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = str(cuda_n)
import torch
torch.cuda.set_device(cuda_n)
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
from inference import *
from networks import *
from init_methods import *
from pyro.nn import AutoRegressiveNN
from gmms import *

"""
Initialize Hyperparameters
"""

d = 50 #latent dim
batch_size = 128
learning_rate = 3e-4
num_epochs = 0
stop_early= False
binary_data = False
K=1
valid_size = 0.1
num_epochs_test = 300

#results="/home/sakshia1/myresearch/missing_dasta/miwae/pytorch/results/mnist-" + str(binary_data) + "-"
##results for beta-annealing
#results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-beta-annealing-"
##Results for alpha-annealing
results=os.getcwd() + "/results/svhn/" 
ENCODER_PATH = "models/svhn_encoder_anneal.pth"  ##without 20 is d=50
DECODER_PATH = "models/svhn_decoder_anneal.pth"  ##simple is for simple VAE

"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""

train_loader, val_loader = train_valid_loader_svhn(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data)

#train_loader, val_loader = train_valid_loader_svhn(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data)
"""
Initialize the network and the Adam optimizer
"""

channels = 3    #1 for MNist
p = 32          # 28 for mnist
q = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

#encoder =  FlatWideResNet(channels=channels, size=2, levels=3, blocks_per_level=4, out_features = 2*d, shape=(p,q))
#decoder = FlatWideResNetUpscaling(channels=channels, size=2, levels=3, blocks_per_level=4, in_features = d, shape=(p,q), model ='sigma_vae')

encoder =  FlatWideResNet(channels=channels, size=2, levels=3, dense_blocks=2, out_features = 2*d, activation=nn.LeakyReLU(), shape=(p,q)) #blocks_per_level=2,
decoder = FlatWideResNetUpscaling(channels=channels, size=2, levels=3, dense_blocks=2, transpose=True, activation=nn.LeakyReLU(), in_features = d, shape=(p,q), model ='sigma_vae')

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
    encoder, decoder = train_VAE_SVHN(num_epochs, train_loader, val_loader, ENCODER_PATH, DECODER_PATH, results, encoder, decoder, optimizer, p_z, device, d, stop_early, annealing=True)

### Load model 
checkpoint = torch.load(ENCODER_PATH, map_location='cuda:1')
#print(checkpoint)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH, map_location='cuda:1')
#checkpoint['log_sigma'] = checkpoint['log_sigma'].unsqueeze(0)
#print(decoder)
decoder.load_state_dict(checkpoint['model_state_dict'])
print(torch.cuda.current_device())
print("model loaded")

encoder.eval()
decoder.eval()

file_save = results + str(-1) + "/gmms_svhn.pkl"

do_training = False

if do_training:
    if os.path.exists(file_save):
        with open(file_save, 'rb') as file:
            gm = pickle.load(file)
    else:
        train_gaussian_mixture(train_loader, encoder, d, batch_size, results, file_save, data_='svhn')

for params in encoder.parameters():
    params.requires_grad = False

for params in decoder.parameters():
    params.requires_grad = False

burn_in_period = 20

#mixture_loss = np.zeros((6,10,num_epochs_test))
#mixture_mse = np.zeros((6,10,num_epochs_test))

#print(decoder)
###Generate 500 samples from decoder
#for i in range(100):
#    x = generate_samples(p_z, decoder, d, L=1, data='svhn').cpu().data.numpy().reshape(1,3,32,32)  
#    plot_image_svhn(np.squeeze(x), os.getcwd() + "/results/generated-samples/" + str(i)+ ".png" ) 

#xm_loss = np.zeros((6,10,num_epochs_test))
#xm_loss_per_img = np.zeros((6,10,num_epochs_test))
#xm_mse_per_img = np.zeros((6,10,num_epochs_test))
#upper_bound_ = np.zeros((10,num_epochs_test))
#lower_bound_ = np.zeros((10,num_epochs_test))
#num_images = 10
#z_kl_loss = np.zeros((6,10,num_epochs_test))
#z_loglike_loss = np.zeros((6,10,num_epochs_test))
#xm_mse = np.zeros((6,10,num_epochs_test))
#iaf_loss = np.zeros((6,10,num_epochs_test))
#iaf_mse = np.zeros((6,10,num_epochs_test))
#z_loss = np.zeros((6,10,num_epochs_test))
#z_mse = np.zeros((6,10,num_epochs_test))
#xm_loss_NN = np.zeros((6,10,num_epochs_test))
#xm_mse_NN = np.zeros((6,10,num_epochs_test))
#term1 = np.zeros((6, 10, num_epochs_test))  #loglikelihood
#term2 = np.zeros((6, 10, num_epochs_test))  #KL
#term3 = np.zeros((6, 10, num_epochs_test))  #Entropy

xm_loss = np.zeros((num_epochs_test))
xm_mse = np.zeros((num_epochs_test))
iaf_loss = np.zeros((num_epochs_test))
iaf_mse = np.zeros((num_epochs_test))
z_loss = np.zeros((num_epochs_test))
z_mse = np.zeros((num_epochs_test))
mixture_loss = np.zeros((num_epochs_test))
mixture_mse = np.zeros((num_epochs_test))
mixture_loss_inits =  np.zeros((num_epochs_test))
xm_loss_NN = np.zeros((num_epochs_test))
xm_mse_NN = np.zeros((num_epochs_test))
iaf_gaussian_loss = np.zeros((num_epochs_test))
iaf_mixture_loss =  np.zeros((num_epochs_test))
iaf_mixture_reinits_loss =  np.zeros((num_epochs_test))

K_samples = 100

print(torch.cuda.current_device())

for iterations in range(1):
    for i in [-1]:
        p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
        p_z_eval = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

        ### Get test loader for different missing percentage value
        #print("memory before --" )
        #print(torch.cuda.memory_allocated(device=0))
        ## MAR (0.5, 0.8)
        #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = i),batch_size=1)
        ## Right half missing (0)
        #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, top_half=True),batch_size=1)
        ## 4 patches of size 10*10 missing (-1)
        test_loader = torch.utils.data.DataLoader(dataset=SVHN_Test(top_half=True),batch_size=1)
        
        print("test data loaded")
        test_log_likelihood, test_loss, test_mse, nb, = 0, 0, 0, 0
        start = datetime.now()
        #np.random.seed(5678)
        ### Optimize over test variational parameters

        ##Init xm_params with output of decoder:
        #pseudo_gibbs_loss = np.zeros((num_epochs_test))
        #pseudo_gibbs_mse = np.zeros((num_epochs_test))
        #pseudo_gibbs_elbo = np.zeros((num_epochs_test))
        #pseudo_gibbs_loglike = np.zeros((num_epochs_test))
        #m_elbo = np.zeros((num_epochs_test))
        #m_mse = np.zeros((num_epochs_test))
        #m_loss = np.zeros((num_epochs_test))
        #m_loglikelihood = np.zeros((num_epochs_test))
        #xm_elbo = np.zeros((num_epochs_test))
        #xm_loglikelihood = np.zeros((num_epochs_test))
        #lower_bound_all = upper_bound_all = lower_log_like_all = upper_log_like_all = lower_err = upper_err = 0
        #total_loss = 0

        lower_bound = upper_bound = bound_updated_encoder = bound_updated_test_encoder = pseudo_iwae = m_iwae = xm_iwae = xm_NN_iwae = iaf_iwae = z_iwae = mixture_iwae = mixture_iwae_inits = 0
        total_loss = 0

        num_images_to_run = 1000
        pseudo_gibbs_sample = []
        metropolis_gibbs_sample = []
        iaf_params = []
        z_params = []
        mixture_params_inits = []
        mixture_params = []
        iaf_gaussian_params = []
        iaf_mixture_params = []
        iaf_mixture_params_re_inits = []

        print(device)

        nb = 0

        for data in test_loader:
            nb += 1
            if nb==1001:
                break
            #if nb<71:
            #    continue
            
            print("Image : ", nb)
            b_data, b_mask, b_full = data
            channels = b_data.shape[1]
            p = b_data.shape[2]
            q = b_data.shape[3]

            b_data_init = b_data
            missing_counts = channels*p*q - int(b_mask.sum())

            b_mask = b_mask.to(device,dtype = torch.bool)
            b_full_ = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float)
            
            #plot_image_svhn(np.squeeze(b_data.cpu().data.numpy()),results  + "generated-images/true_image.png")

            burn_in_ = True
            random = False

            img = b_full.cpu().data.numpy()         ## added .data
            plot_image_svhn(np.squeeze(img),results + str(i) + "/compiled/" + str(nb%10) +  "true.png")

            missing = b_data
            missing[~b_mask] = 0.5      
            img = missing.cpu().data.numpy() 
            plot_image_svhn(np.squeeze(img),results + str(i) + "/compiled/" + str(nb%10) +   "missing.png" )

            b_data_low = b_data
            b_data_sample = b_data
            b_data_low[~b_mask] = 0.5
            x_logits_init = torch.zeros_like(b_data)
            p_xm = td.Normal(loc = 0.5 + x_logits_init[~b_mask].reshape([-1,1]), scale =  torch.ones_like(b_data)[~b_mask].reshape([-1,1])) #.to(device,dtype = torch.float)
            b_data_sample[~b_mask] = p_xm.sample().reshape(-1)  

            channel_0 = torch.mean(b_data[:,0,:,:][b_mask[:,0,:,:]])
            channel_1 = torch.mean(b_data[:,1,:,:][b_mask[:,1,:,:]])
            channel_2 = torch.mean(b_data[:,2,:,:][b_mask[:,2,:,:]])
            b_data[:,0,:,:][~b_mask[:,0,:,:]] = channel_0
            b_data[:,1,:,:][~b_mask[:,1,:,:]] = channel_1
            b_data[:,2,:,:][~b_mask[:,2,:,:]] = channel_2
        
            lower_bound +=  eval_iwae_bound(iota_x = b_data.to(device,dtype = torch.float), full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=K_samples, data='svhn')
            upper_bound +=  eval_iwae_bound(iota_x = b_full.to(device,dtype = torch.float), full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=K_samples, data='svhn')

            imputation = b_data.to(device,dtype = torch.float)
            xm_logits, out_encoder, sigma_decoder = mvae_impute_svhn(b_data.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, L=1)
            imputation[~b_mask] = xm_logits[~b_mask]
            plot_image_svhn(np.squeeze(imputation.cpu().data.numpy()), results + str(i) + "/compiled/"  + str(nb%10) + "-miss-imputation.png" )

            #imputation_ = b_data.to(device,dtype = torch.float)
            #xm_logits, out_encoder, sigma_decoder = mvae_impute_svhn( b_full.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, L=1)
            #imputation_[~b_mask] = xm_logits[~b_mask]
            #plot_image_svhn(np.squeeze(imputation_.cpu().data.numpy()), results + str(i) + "/compiled/"  + "-true-imputation.png" )

            print("Lower IWAE bound (0's) : ", lower_bound)
            print("Upper IWAE bound (true image) : ", upper_bound)

            if random: 
                x_logits_init = torch.zeros_like(b_data)
                p_x_m = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_init[~b_mask]),1)
                b_data_init = b_data
                b_data_init[~b_mask] = p_x_m.sample()
                x_logits_init[~b_mask] = torch.log(b_data_init/(1-b_data_init))[~b_mask]
            elif burn_in_:
                x_logits_init, b_data_init, z_init = burn_in_svhn(b_data.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, burn_in_period=burn_in_period, data='svhn')
                x_logits_init = x_logits_init.cpu()
                b_data_init = b_data_init.cpu()
                #b_data_init = b_data
                #b_data_init[~b_mask] = k_neighbors_sample(b_data, b_mask) 
                #b_data_init = b_data_init
                #x_logits_init = torch.log(b_data_init/(1-b_data_init))

            sampled_image = b_data_init
            sampled_image_m = sampled_image
            sampled_image_o = sampled_image

            if iterations==-1:
                sampled_image[~b_mask] = b_full[~b_mask].to('cpu',dtype = torch.float)        
                sampled_image_m[~b_mask] = b_full[~b_mask].to('cpu',dtype = torch.float)       
                z_init =  encoder.forward(b_full.to(device,dtype = torch.float))
                sampled_image_o = sampled_image_m
            else:
                #sampled_image = b_data_init
                sampled_image_m = sampled_image
                sampled_image_o = sampled_image
                z_init =  encoder.forward(b_data_init.to(device,dtype = torch.float))

            if not burn_in:
                plot_image_svhn(np.squeeze(sampled_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "init.png" )
            else:
                if iterations==-1:
                    plot_image_svhn(np.squeeze(sampled_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "-burn-in.png" )
                else:
                    burn_in_image = b_data.to(device,dtype = torch.float)
                    burn_in_image[~b_mask] = x_logits_init[~b_mask].to(device,dtype = torch.float)
                    plot_image_svhn(np.squeeze(burn_in_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "-burn-in.png" )
         
            start_pg = datetime.now()
            x_logits_pseudo_gibbs, x_sample_pseudo_gibbs, iwae, sample  = pseudo_gibbs(sampled_image.to(device,dtype = torch.float), b_data.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, results, iterations, T=num_epochs_test, nb=nb, K = 1, data='svhn', full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float))
            end_pg = datetime.now()
            diff_z = end_pg - start_pg
            print(" Time taken for pseudo-gibbs", diff_z.total_seconds())
            pseudo_gibbs_sample.append(sample)
            pseudo_iwae += iwae

            #Impute image with pseudo-gibbs
            pseudo_gibbs_image = b_data.to(device,dtype = torch.float)
            pseudo_gibbs_image[~b_mask] = x_logits_pseudo_gibbs[~b_mask]
            plot_image_svhn(np.squeeze(pseudo_gibbs_image.cpu().data.numpy()), results + str(i) + "/compiled/" +  str(nb%10) + "pseudo-gibbs.png" )

            ##M-with-gibbs sampler
            start_m = datetime.now()
            m_nelbo, m_error, x_full_logits, m_loglike, iwae, sample =  m_g_sampler(iota_x = sampled_image_m.to(device,dtype = torch.float), full = b_full.to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, results = results, nb = nb, iterations = iterations, K= 1, T=num_epochs_test,data='svhn')
            end_m = datetime.now()
            diff_z = end_m - start_m
            print(" Time taken for metropolis within gibbs", diff_z.total_seconds())
        
            metropolis_gibbs_sample.append(sample)
            m_iwae +=iwae

            #Impute image with metropolis-within-pseudo-gibbs
            metropolis_image = b_data.to(device,dtype = torch.float)
            metropolis_image[~b_mask] = x_full_logits[~b_mask]
            plot_image_svhn(np.squeeze(metropolis_image.cpu().data.numpy()), results + str(i) + "/compiled/" + str(nb%10) + "metropolis-within-pseudo-gibbs.png" )

            dd = False

            if dd:
                if iterations==-1:
                    xm_params = b_full[~b_mask]
                else:
                    xm_params = x_logits_init[~b_mask]

                ##Optimize q(x_m) with burn-in
                start_o = datetime.now()

                scales = torch.ones(*xm_params.shape).cuda()

                ##Change for svhn
                xm_nelbo_, xm_error_ = optimize_q_xm(num_epochs = num_epochs_test, xm_params = xm_params, z_params = z_init, b_data = b_data.to(device,dtype = torch.float), sampled_image_o = sampled_image_o.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), b_full = b_full.to(device,dtype = torch.float), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb= nb, file = "q_xm-all.png", p_z = p_z, K_samples = K_samples, data='svhn', scales=scales )
                end_o = datetime.now()
                diff_o = end_o - start_o
                print(" Time taken for optimizing q(x_m)", diff_o.total_seconds())
                
                xm_loss[iterations, nb%10, : ] = xm_nelbo_
                xm_mse[iterations, nb%10, : ] = xm_error_

                ##Optimize q(x_m) with nearest-neighbour 
                b_data_init_ = b_data
                #b_data_init_ = b_data_init_.to(device,dtype = torch.float)
                b_data_init_[~b_mask] = k_neighbors_svhn(b_data, b_mask) 
                x_logits_init_ = torch.log(b_data_init_/(1-b_data_init_))

                plot_image_svhn(np.squeeze(b_data_init_.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + '-' + "NN-init.png" )
                prefix = results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + '-'
                plot_images_in_row(num_epochs = num_epochs_test, loc1 = results + str(i) + "/images/" + str(nb%10) + "/"  +  "true.png" , loc2 = results + str(i) + "/images/" + str(nb%10) + "/"  +  "missing.png" , loc3 = prefix + "burn-in.png" , loc4 = prefix + "NN-init.png", loc5 = prefix + "NN-init.png" , file = prefix + "image-inits.png")

                if iterations==-1:
                    xm_params = b_full[~b_mask]
                else:
                    xm_params = x_logits_init[~b_mask]
                    #print("")
                    #print("x-params from KNN ------", xm_params)
                scales = torch.ones(*xm_params.shape).cuda()

                gc.collect()
                start_o = datetime.now()
                xm_nelbo_, xm_error_ = optimize_q_xm(num_epochs = num_epochs_test, xm_params = xm_params, z_params = z_init, b_data = b_data.to(device,dtype = torch.float), sampled_image_o = sampled_image_o.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), b_full = b_full.to(device,dtype = torch.float), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb = nb, file = "q_xm-all-NN.png" , p_z = p_z, K_samples = K_samples, data='svhn', scales= scales  )
                end_o = datetime.now()
                diff_o = end_o - start_o
                print(" Time taken for optimizing q(x_m)", diff_o.total_seconds())

                xm_loss_NN[iterations, nb%10, : ] = xm_nelbo_
                xm_mse_NN[iterations, nb%10, : ] = xm_error_

                b_data[~b_mask] = 0

            if iterations==-1:
                z_init =  encoder.forward(b_full.to(device,dtype = torch.float))
            else:
                z_init =  encoder.forward(b_data.to(device,dtype = torch.float))

            start_iaf = datetime.now()
            channel_0 = torch.mean(b_data[:,0,:,:][b_mask[:,0,:,:]])
            channel_1 = torch.mean(b_data[:,1,:,:][b_mask[:,1,:,:]])
            channel_2 = torch.mean(b_data[:,2,:,:][b_mask[:,2,:,:]])
            b_data[:,0,:,:][~b_mask[:,0,:,:]] = channel_0
            b_data[:,1,:,:][~b_mask[:,1,:,:]] = channel_1
            b_data[:,2,:,:][~b_mask[:,2,:,:]] = channel_2


            xm_nelbo_, xm_error_, iwae, t1, t2, params_ = optimize_IAF(num_epochs = num_epochs_test, z_params = z_init, b_data = b_data.to(device,dtype = torch.float), sampled_image_o = sampled_image_o.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), b_full = b_full.to(device,dtype = torch.float), p_z = p_z, encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb, K_samples = K_samples, data='svhn', p_z_eval = p_z_eval, with_gaussian=True )
            end_iaf = datetime.now()
            diff_iaf = end_iaf - start_iaf
            print("Time taken for optimizing IAF : ", diff_iaf.total_seconds())
            #print(t1.state_dict())
            iaf_params.append([t1.state_dict(), t2.state_dict(), params_])
            #print(t1.state_dict())
            iaf_iwae += iwae
            iaf_loss += xm_nelbo_
            iaf_mse += xm_error_

            #b_data[~b_mask] = 0
            if iterations==-1:
                z_init =  encoder.forward(b_full.to(device,dtype = torch.float))
            else:
                z_init =  encoder.forward(b_data.to(device,dtype = torch.float))

            start_z = datetime.now()
            z_nelbo_, z_error_ , iwae, params = optimize_z(num_epochs = num_epochs_test, p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float),  b_mask = b_mask.to(device,dtype = torch.bool),  encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples, data='svhn', p_z_eval = p_z_eval )
            end_z = datetime.now()
            diff_z = end_z - start_z
            print("Time taken for optimizing z : ", diff_z.total_seconds())
            z_iwae += iwae
            z_loss  += z_nelbo_
            z_mse += z_error_
            z_params.append(params)
            #print(z_params)
            #print(torch.cuda.current_device())
            #means_ = torch.from_numpy(gm.means_)
            #std_ = torch.sqrt(torch.from_numpy(gm.covariances_))
            #weights_ = torch.from_numpy(gm.weights_)
            #p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(probs=weights_.cuda()), td.Independent(td.Normal(means_.cuda(), std_.cuda()), 1))

            #With re-inits
            start_mix = datetime.now()
            z_nelbo_, z_error_, iwae, logits, means, scales = optimize_mixture(num_epochs = num_epochs_test, p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float),  b_mask = b_mask.to(device,dtype = torch.bool), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples, data='svhn')
            end_mix = datetime.now()
            diff_mix = end_mix - start_mix
            print("Time taken for optimizing mixtures (re-inits): ", diff_mix.total_seconds())
            mixture_iwae_inits += iwae
            mixture_loss_inits += z_nelbo_
            mixture_mse += z_error_
            mixture_params_inits.append([logits, means, scales])

            #print(logits, means, scales)
            #Without re-inits
            start_mix = datetime.now()

            z_nelbo_, z_error_, iwae, logits, means, scales = optimize_mixture(num_epochs = num_epochs_test, p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float),  b_mask = b_mask.to(device,dtype = torch.bool), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples, data='svhn', do_random=False)
            end_mix = datetime.now()
            diff_mix = end_mix - start_mix
            print("Time taken for optimizing mixtures : ", diff_mix.total_seconds())
            mixture_iwae +=iwae
            mixture_loss += z_nelbo_
            mixture_params.append([logits, means, scales])
            
            prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 

            prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 

            print(lower_bound/nb, upper_bound/nb, bound_updated_encoder/nb, pseudo_iwae/nb, m_iwae/nb,  xm_iwae/nb,  xm_NN_iwae/nb,  iaf_iwae/nb, z_iwae/nb, mixture_iwae/nb, mixture_iwae_inits/nb) #added mixture_iwae_inits later

            file_save_params = results + str(-1) + "/pickled_files/params_svhn_TH.pkl"

            with open(file_save_params, 'wb') as file:
                pickle.dump([pseudo_gibbs_sample,metropolis_gibbs_sample,z_params,iaf_params, mixture_params_inits,mixture_params,nb], file)
            #pickle.dump([iaf_gaussian_params, iaf_mixture_params ,iaf_mixture_params_re_inits ,nb], file)

            file_loss = results + str(-1) + "/pickled_files/loss_svhn_TH.pkl"
            with open(file_loss, 'wb') as file:
                pickle.dump([z_loss, mixture_loss_inits, mixture_loss, iaf_loss,  nb], file)
                #pickle.dump([iaf_gaussian_loss, iaf_mixture_loss, iaf_mixture_reinits_loss, nb], file)

            ##For svhn
            #plot_all_images(i, nb, iterations, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'image-comparison.png', prefix, data='svhn')
            gc.collect()
  

do_plot= False

if do_plot:
    plot_5runs(num_epochs_test, xm_loss, iaf_loss, z_loss, xm_loss_NN, mixture_loss, results, -1)

    #compare_ELBO(num_epochs_test, xm_loss[:,1,:], iaf_loss[:,1,:], z_loss[:,1,:], xm_loss_NN[:,1,:], mixture_loss[:,1,:], results, -1, ylim1=-3000, ylim2=0)

    for image in range(10):
        value = mixture_loss[0,image,-1]
        print(image)
        compare_ELBO(num_epochs_test, xm_loss[:,image,:], iaf_loss[:,image,:], z_loss[:,image,:], xm_loss_NN[:,image,:], mixture_loss[:,image,:] , results, -1, ylim2 = 0, image = image) #ylim1=value - 10, ylim2=value + 20,



