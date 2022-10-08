cuda_n=0
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
from evaluate_helper import *
"""
Initialize Hyperparameters
"""

d = 50 #latent dim
batch_size = 128
learning_rate = 6e-4 #3e-4
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
##with lower threshold of std
#ENCODER_PATH = "models/svhn_encoder_"  ##without 20 is d=50
#DECODER_PATH = "models/svhn_decoder_"  ##simple is for simple VAE
#ENCODER_PATH = "models/svhn_encoder_stop_early" 
#DECODER_PATH = "models/svhn_decoder_stop_early" 
##with truncated normal
ENCODER_PATH = "models/svhn_encoder_TN_stop_early"  ##without 20 is d=50
DECODER_PATH = "models/svhn_decoder_TN_stop_early"    ##simple is for simple VAE

##The following encoder/decoder pair is for -1,1 data
#ENCODER_PATH = "models/svhn_encoder_anneal_norm_0-1_stop_early"  ##without 20 is d=50
#DECODER_PATH = "models/svhn_decoder_anneal_norm_0-1_stop_early"  ##simple is for simple VAE
ENCODER_PATH_UPDATED  =  "models/svhn_encoder_TN_TH-updated.pt.pth"  
ENCODER_PATH_UPDATED_TEST = "models/svhn_encoder_TN_TH-updated_test_1000.pt.pth"

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
encoder_updated =  FlatWideResNet(channels=channels, size=2, levels=3, dense_blocks=2, out_features = 2*d, activation=nn.LeakyReLU(), shape=(p,q)) #blocks_per_level=2,
encoder_updated_test =  FlatWideResNet(channels=channels, size=2, levels=3, dense_blocks=2, out_features = 2*d, activation=nn.LeakyReLU(), shape=(p,q)) #blocks_per_level=2,

encoder = encoder.cuda()
decoder = decoder.cuda()
encoder_updated = encoder_updated.cuda()
encoder_updated_test = encoder_updated_test.cuda()
print(torch.cuda.current_device())

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""

p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

best_loss, count = 0, 0

if num_epochs>0:
    encoder, decoder = train_VAE_SVHN(num_epochs, train_loader, val_loader, ENCODER_PATH,  results, encoder, decoder, optimizer, p_z, device, d, stop_early, annealing=True, DECODER_PATH= DECODER_PATH)

### Load model 
checkpoint = torch.load(ENCODER_PATH + ".pth", map_location='cuda:0')
#print(checkpoint)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH + ".pth", map_location='cuda:0')
#checkpoint['log_sigma'] = checkpoint['log_sigma'].unsqueeze(0)
#print(decoder)
decoder.load_state_dict(checkpoint['model_state_dict'])
print(torch.cuda.current_device())
print("model loaded")

checkpoint = torch.load(ENCODER_PATH_UPDATED , map_location='cuda:0')
encoder_updated.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(ENCODER_PATH_UPDATED_TEST , map_location='cuda:0')
encoder_updated_test.load_state_dict(checkpoint['model_state_dict'])

for params in encoder.parameters():
    params.requires_grad = False

for params in decoder.parameters():
    params.requires_grad = False

for params in encoder_updated.parameters():
    params.requires_grad = False

for params in encoder_updated_test.parameters():
    params.requires_grad = False
    
encoder.eval()
decoder.eval()
encoder_updated.eval()
encoder_updated_test.eval()

file_save = os.getcwd()  + "/models/gmms_svhns.pkl" ##change for gabe

do_training = False

if do_training:
    if os.path.exists(file_save):
        with open(file_save, 'rb') as file:
            gm = pickle.load(file)
    else:
        train_gaussian_mixture(train_loader, encoder, d, batch_size, results, file_save, data_='svhn')

burn_in_period = 20

print((torch.nn.Softplus()(decoder.get_parameter("log_sigma"))))
#mixture_loss = np.zeros((6,10,num_epochs_test))
#mixture_mse = np.zeros((6,10,num_epochs_test))
#print(decoder)
###Generate 500 samples from decoder
for i in range(100):
    x = generate_samples(p_z, decoder, d, L=1, data='svhn').cpu().data.numpy().reshape(1,3,32,32)  
    plot_image_svhn(np.squeeze(x), os.getcwd() + "/results/generated-samples/" + str(i)+ ".png" ) 
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

K_samples = 1000

print(torch.cuda.current_device())

for iterations in range(1):
    for i in [-1]:
        p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
        p_z_eval = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

        #means_ = torch.from_numpy(gm.means_)
        #std_ = torch.sqrt(torch.from_numpy(gm.covariances_))
        #weights_ = torch.from_numpy(gm.weights_)
        #p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(probs=weights_.cuda()), td.Independent(td.Normal(means_.cuda(), std_.cuda()), 1))
        #p_z_eval = td.mixture_same_family.MixtureSameFamily(td.Categorical(probs=weights_.cuda()), td.Independent(td.Normal(means_.cuda(), std_.cuda()), 1))

        ### Get test loader for different missing percentage value
        #print("memory before --" )
        #print(torch.cuda.memory_allocated(device=0))
        ## MAR (0.5, 0.8)
        #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = i),batch_size=1)
        ## Right half missing (0)
        #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, top_half=True),batch_size=1)
        ## 4 patches of size 10*10 missing (-1)
        test_loader = torch.utils.data.DataLoader(dataset=SVHN_Test(top_half=True),batch_size=1) # patches top_half
        
        print("test data loaded")
        test_log_likelihood, test_loss, test_mse, nb, = 0, 0, 0, 0
        start = datetime.now()
        #np.random.seed(5678)
        ### Optimize over test variational parameters
        max_samples = 10000

        # =  =  =  =  =  = xm_iwae = xm_NN_iwae =  =  = mixture_iwae = mixture_iwae_inits =  = 
        bound_encoder_test = np.zeros((max_samples))
        bound_encoder_train = np.zeros((max_samples))
        upper_bound = np.zeros((max_samples))
        lower_bound = np.zeros((max_samples))
        
        pseudo_iwae = np.zeros((max_samples))
        m_iwae = np.zeros((max_samples))
        iaf_iwae = np.zeros((max_samples))
        z_iwae = np.zeros((max_samples))
        mixture_iwae = np.zeros((max_samples))
        mixture_iwae_inits = np.zeros((max_samples))        
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
            if nb==501:
                break
            #if nb<20:
            #    continue
            images = []
            print("Image : ", nb)
            b_data, b_mask, b_full = data
            
            channels = b_data.shape[1]
            p = b_data.shape[2]
            q = b_data.shape[3]
            missing_counts = channels*p*q - int(b_mask.sum())

            b_mask_cpu = b_mask.cpu().data.numpy()
            b_data_cpu = b_data.cpu().data.numpy()
            b_mask = b_mask.to(device,dtype = torch.bool)
            b_full_ = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float)
            print("data --",torch.max(b_full_[~b_mask]), torch.min(b_full_[~b_mask]) )
            #plot_image_svhn(np.squeeze(b_full_.cpu().data.numpy()),results  + str(i) + "/compiled/" + str(nb%10) +   "true.png")
            images.append(np.squeeze(b_full_.cpu().data.numpy()))
            
            burn_in_ = True
            random = False
            
            b_data_burn = mean_impute(b_data.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool))
        
            images.append(np.squeeze(b_data_burn.cpu().data.numpy()))
            #lower_bound +=  eval_iwae_bound(iota_x = b_data.to(device,dtype = torch.float), full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=10000, data='svhn')
            lower_bound += eval_baseline(max_samples, p_z, encoder, decoder, b_data_burn.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn')

            #bound_encoder_train +=  eval_iwae_bound(iota_x = b_data.to(device,dtype = torch.float), full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder_updated,decoder = decoder, p_z= p_z, d=d, K=10000, data='svhn')

            bound_encoder_train += eval_baseline(max_samples, p_z, encoder_updated, decoder, b_data_burn.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn', nb=nb)
            
            #bound_encoder_test +=  eval_iwae_bound(iota_x = b_data.to(device,dtype = torch.float), full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder_updated_test,decoder = decoder, p_z= p_z, d=d, K=10000, data='svhn')
            bound_encoder_test += eval_baseline(max_samples, p_z, encoder_updated_test, decoder, b_data_burn.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn', nb=nb, test=True)

            #upper_bound +=  eval_iwae_bound(iota_x = b_full.to(device,dtype = torch.float), full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=10000, data='svhn')
            upper_bound += eval_baseline(max_samples, p_z, encoder, decoder, b_full.to(device,dtype = torch.float), b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, data='svhn')

            imputation = decoder_impute(b_data_burn.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), encoder,  decoder,  p_z, d, L=1 )

            print(torch.sum(b_data_burn[~b_mask]), np.sum(b_data_cpu[~b_mask_cpu]), np.sum(imputation[~b_mask_cpu]))

            images.append(np.squeeze(imputation))
            
            imputation = decoder_impute(b_data_burn.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), encoder_updated,  decoder,  p_z, d, L=1 )
            images.append(np.squeeze(imputation))
            print(torch.sum(b_data_burn[~b_mask]), np.sum(b_data_cpu[~b_mask_cpu]), np.sum(imputation[~b_mask_cpu]))
            
            imputation = decoder_impute(b_data_burn.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), encoder_updated_test,  decoder,  p_z, d, L=1 )
            images.append(np.squeeze(imputation))
            print(torch.sum(b_data_burn[~b_mask]), np.sum(b_data_cpu[~b_mask_cpu]), np.sum(imputation[~b_mask_cpu]))

            plot_images_svhn_block(images, file=results + str(i) + "/compiled/"  + str(nb%10) + "baselines.png")
            print("Lower IWAE bound (0's) : ", lower_bound[-1]/nb)
            print("Upper IWAE bound (true image) : ", upper_bound[-1]/nb)
            print("IWAE bound (0's + encoder updated) : ", bound_encoder_train[-1]/nb)
            print("IWAE bound (0's + encoder updated) : ", bound_encoder_test[-1]/nb)

            #With re-inits
            start_mix = datetime.now()
            z_nelbo_, z_error_, iwae, logits, means, scales = optimize_mixture(num_epochs = num_epochs_test, p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float),  b_mask = b_mask.to(device,dtype = torch.bool), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples, data='svhn')
            end_mix = datetime.now()
            diff_mix = end_mix - start_mix
            print("Time taken for optimizing mixtures (re-inits): ", diff_mix.total_seconds())
            #mixture_iwae_inits += iwae
            mixture_loss_inits += z_nelbo_
            mixture_mse += z_error_

            mixture_params_inits.append([logits, means, scales])
            mixture_iwae_inits +=  evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = [logits, means, scales], decoder = decoder, device = device, d = d, results = results, nb=i , K_samples = max_samples, ismixture=True, data='svhn', do_random=True )
            print("IWAE for mixture (re-inits) : ", mixture_iwae_inits[-1]/nb)

            #Without re-inits
            start_mix = datetime.now()
            z_nelbo_, z_error_, iwae, logits, means, scales = optimize_mixture(num_epochs = num_epochs_test, p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float),  b_mask = b_mask.to(device,dtype = torch.bool), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples, data='svhn', do_random=False)
            end_mix = datetime.now()
            diff_mix = end_mix - start_mix
            print("Time taken for optimizing mixtures : ", diff_mix.total_seconds())
            #mixture_iwae +=iwae
            mixture_loss += z_nelbo_
            mixture_params.append([logits, means, scales])
 
            mixture_iwae +=  evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = [logits, means, scales], decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, ismixture=True, data='svhn' )
            print("IWAE for mixture : ", mixture_iwae[-1]/nb)

            start_iaf = datetime.now()
            xm_nelbo_, xm_error_, iwae, t1, t2, params  = optimize_IAF(num_epochs = num_epochs_test,  b_data = b_data_burn.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), b_full = b_full.to(device,dtype = torch.float), p_z = p_z, encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb, K_samples = K_samples, data='svhn', p_z_eval = p_z_eval, with_gaussian=False ) #, params_
            end_iaf = datetime.now()
            diff_iaf = end_iaf - start_iaf
            print("Time taken for optimizing IAF : ", diff_iaf.total_seconds())
            #print(t1.state_dict())
            iaf_params.append([t1.state_dict(), t2.state_dict(), params]) #, params_
            #print(t1.state_dict())
            #iaf_iwae += iwae
            iaf_loss += xm_nelbo_
            iaf_mse += xm_error_  

            iaf_iwae += evaluate_iaf(p_z = p_z, b_data = b_data.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),iaf_params = [t1.state_dict(), t2.state_dict(), params], encoder = encoder, decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, data='svhn' )
            print("IWAE for IAF : ", iaf_iwae[-1]/nb)

            start_z = datetime.now()
            z_nelbo_, z_error_ , iwae, params = optimize_z(num_epochs = num_epochs_test, p_z = p_z, b_data = b_data_burn.to(device,dtype = torch.float), b_full = b_full.to(device,dtype = torch.float),  b_mask = b_mask.to(device,dtype = torch.bool),  encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples, data='svhn', p_z_eval = p_z_eval )
            end_z = datetime.now()
            diff_z = end_z - start_z
            print("Time taken for optimizing z : ", diff_z.total_seconds())
            #z_iwae += iwae
            z_loss  += z_nelbo_
            z_mse += z_error_
            z_params.append(params)

               
            z_iwae += evaluate_z(p_z = p_z, b_data = b_data.to(device,dtype = torch.float),  b_full = b_full.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),z_params = params.to(device,dtype = torch.float), decoder = decoder, device = device, d = d, results = results, nb=nb , K_samples = max_samples, data='svhn' )
            print("IWAE for gaussian : ", z_iwae[-1])

            #display_images_svhn_all(decoder, z_params[nb-1], b_data.to(device,dtype = torch.float), encoder, iaf_params[nb-1], mixture_params[nb-1], mixture_params_inits[nb-1], d, results  + str(-1) + "/compiled/" + str(nb%10)  +'optimization_methods.png', k = 20)
               
            #print("in main file --", b_data_burn[:,1,:,:][~b_mask[:,1,:,:]])
            plot_image_svhn(np.squeeze(b_data_burn.cpu().data.numpy()), results + str(i) + "/compiled/"  + "-mean-values.png" )
            
            if random: 
                x_logits_init = torch.zeros_like(b_data)
                p_x_m = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_init[~b_mask]),1)
                b_data_init = b_data
                b_data_init[~b_mask] = p_x_m.sample()
                x_logits_init[~b_mask] = torch.log(b_data_init/(1-b_data_init))[~b_mask]
            elif burn_in_:
                print(torch.mean(b_full[0,0][b_mask[0,0]]), torch.mean(b_full[0,1][b_mask[0,1]]), torch.mean(b_full[0,2][b_mask[0,2]]))
                x_logits_init, b_data_init, z_init = burn_in_svhn(b_data_burn.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, burn_in_period=burn_in_period, data='svhn')
                x_logits_init = x_logits_init.cpu()
                b_data_init = b_data_init.cpu()
                #b_data_init = b_data
                #b_data_init[~b_mask] = k_neighbors_sample(b_data, b_mask) 
                #b_data_init = b_data_init
                #x_logits_init = torch.log(b_data_init/(1-b_data_init))

            print("after", torch.mean(b_data))

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
                    burn_in_image = b_data
                    #burn_in_image[~b_mask] = x_logits_init[~b_mask]
                    #plot_image_svhn(np.squeeze(burn_in_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "-burn-in.png" )
         
            start_pg = datetime.now()
            x_logits_pseudo_gibbs, x_sample_pseudo_gibbs, iwae, sample  = pseudo_gibbs_svhn(sampled_image.to(device,dtype = torch.float), b_data.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, results, iterations, T=num_epochs_test, nb=nb, K = 1, data='svhn', full = b_full.reshape([1,channels,p,q]).to(device,dtype = torch.float))
            end_pg = datetime.now()
            diff_z = end_pg - start_pg
            print(" Time taken for pseudo-gibbs", diff_z.total_seconds())
            pseudo_gibbs_sample.append(sample)
            #pseudo_iwae += iwae

            print(torch.mean(b_full[0,0][b_mask[0,0]]), torch.mean(b_full[0,1][b_mask[0,1]]), torch.mean(b_full[0,2][b_mask[0,2]]))
            print("after pseudo-gibbs", torch.mean(b_data))
            #Impute image with pseudo-gibbs
            pseudo_gibbs_image = b_data.cpu().data.numpy()
            x_logits_pseudo_gibbs = x_logits_pseudo_gibbs.cpu().data.numpy()
            pseudo_gibbs_image[~b_mask_cpu] = x_logits_pseudo_gibbs[~b_mask_cpu]
            plot_image_svhn(np.squeeze(pseudo_gibbs_image), results + str(i) + "/compiled/" +  str(nb%10) + "pseudo-gibbs.png")
            
            pseudo_iwae += evaluate_pseudo_gibbs(max_samples, p_z, encoder, decoder, sample.to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device, data='svhn')
            print("IWAE for pseudo-gibbs : ", pseudo_iwae[-1])

            ##M-with-gibbs sampler
            start_m = datetime.now()
            m_nelbo, m_error, x_full_logits, m_loglike, iwae, sample =  m_g_sampler(iota_x = sampled_image_m.to(device,dtype = torch.float), full = b_full.to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, results = results, nb = nb, iterations = iterations, K= 1, T=num_epochs_test,data='svhn')
            end_m = datetime.now()
            diff_z = end_m - start_m
            print(" Time taken for metropolis within gibbs", diff_z.total_seconds())
        
            metropolis_gibbs_sample.append(sample)
            #m_iwae +=iwae

            #Impute image with metropolis-within-pseudo-gibbs
            metropolis_image = b_data.to(device,dtype = torch.float)
            metropolis_image[~b_mask] = x_full_logits[~b_mask]
            plot_image_svhn(np.squeeze(metropolis_image.cpu().data.numpy()), results + str(i) + "/compiled/" + str(nb%10) + "metropolis-within-pseudo-gibbs.png" )
            m_iwae += evaluate_metropolis_within_gibbs(max_samples, p_z, encoder, decoder, sample.to(device,dtype = torch.float), b_data.to(device,dtype = torch.float),  b_full.to(device,dtype = torch.float), b_mask.to(device,dtype = torch.bool), d, device, data='svhn')
            print("IWAE for metropolis within -gibbs : ", m_iwae[-1])

            #print(logits, means, scales)
            
            prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 

            prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 

            print(lower_bound[-1]/nb, upper_bound[-1]/nb, bound_encoder_train[-1]/nb,bound_encoder_test[-1]/nb, pseudo_iwae[-1]/nb, m_iwae[-1]/nb, iaf_iwae[-1]/nb, z_iwae[-1]/nb, mixture_iwae[-1]/nb, mixture_iwae_inits[-1]/nb) #added mixture_iwae_inits later
            file_save_params = results + str(-1) + "/pickled_files/params_svhn_TH_gaussian.pkl"

            with open(file_save_params, 'wb') as file:
                pickle.dump([pseudo_gibbs_sample,metropolis_gibbs_sample,z_params,iaf_params, mixture_params_inits,mixture_params,nb], file)
            #pickle.dump([iaf_gaussian_params, iaf_mixture_params ,iaf_mixture_params_re_inits ,nb], file)

            file_loss = results + str(-1) + "/pickled_files/loss_svhn_TH_gaussian.pkl"
            with open(file_loss, 'wb') as file:
                pickle.dump([z_loss, mixture_loss_inits, mixture_loss, iaf_loss,  nb], file)
                #pickle.dump([iaf_gaussian_loss, iaf_mixture_loss, iaf_mixture_reinits_loss, nb], file)
                
            x = np.arange(max_samples)
            colours = ['g', 'b', 'y', 'r', 'k', 'c']
            compare_iwae(lower_bound/nb, upper_bound/nb, bound_encoder_train/nb, bound_encoder_test/nb, pseudo_iwae/nb, m_iwae/nb, z_iwae/nb, iaf_iwae/nb, mixture_iwae/nb , mixture_iwae_inits/nb,  colours, x, "IWAE", results + str(-1) + "/compiled/IWAEvsSamples_svhn_TH.png", ylim1= -9000, ylim2 = 0)
            
            file_save_iwae = results + str(-1) + "/pickled_files/svhn_TH_infered_iwae_p_gaussian.pkl"
            with open(file_save_iwae, 'wb') as file:
                pickle.dump([lower_bound/nb, upper_bound/nb, bound_encoder_train/nb, bound_encoder_test/nb, pseudo_iwae/nb, m_iwae/nb, z_iwae/nb, iaf_iwae/nb, mixture_iwae/nb , mixture_iwae_inits/nb, nb], file)
                
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



