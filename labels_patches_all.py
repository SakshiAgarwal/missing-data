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
from inference import *
from networks import *
from init_methods import *
from pyro.nn import AutoRegressiveNN
from gmms import *
import pickle

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
num_epochs_test = 500

#results="/home/sakshia1/myresearch/missing_dasta/miwae/pytorch/results/mnist-" + str(binary_data) + "-"
##results for beta-annealing
#results=os.getcwd() + "/results/mnist-" + str(binary_data) + "-beta-annealing-"
##Results for alpha-annealing
results = os.getcwd() + "/results/mnist-" + str(binary_data) + "-"
ENCODER_PATH = "models/e_model_"+ str(binary_data) + "classification.pt"  ##without 20 is d=50
DECODER_PATH = "models/d_model_"+ str(binary_data) + "classification.pt"  ##simple is for simple VAE

DISCRIMINATIVE_PATH = "models/e_model_"+ str(binary_data) + "ygivenx.pt"

"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""

train_loader, val_loader = train_valid_loader(data_dir ="data" , batch_size=1, valid_size = valid_size, binary_data = binary_data, return_labels=True)

#train_loader, val_loader = train_valid_loader_svhn(data_dir ="data" , batch_size=batch_size, valid_size = valid_size, binary_data = binary_data)
"""
Initialize the network and the Adam optimizer
"""

channels = 1 + 10   #1 for MNist
p = 28          # 28 for mnist
q = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

encoder =  FlatWideResNet(channels=channels, size=1, levels=3, blocks_per_level=2, out_features = 2*d, shape=(p,q))
decoder = FlatWideResNetUpscaling(channels=1, size=1, levels=3, blocks_per_level=2, in_features = 10 + d, shape=(p,q))

discriminative_model = FlatWideResNet(channels=1, size=1, levels=3, blocks_per_level=2, out_features = 10, shape=(p,q))

discriminative_model = discriminative_model.cuda()
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
    encoder, decoder = train_VAE(num_epochs, train_loader, val_loader, ENCODER_PATH, DECODER_PATH, results, encoder, decoder, optimizer, p_z, device, d, stop_early, with_labels=True)

### Load model 
checkpoint = torch.load(ENCODER_PATH)
encoder.load_state_dict(checkpoint['model_state_dict'])
checkpoint = torch.load(DECODER_PATH)
decoder.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load(DISCRIMINATIVE_PATH)
discriminative_model.load_state_dict(checkpoint['model_state_dict'])

print(torch.cuda.current_device())
print("model loaded")

##Re-train again, for now we do only the gaussian case
file_save = results + str(-1) + "/classification-gmms-per-class.pkl"

if os.path.exists(file_save):
    with open(file_save, 'rb') as file:
        means,std,weights = pickle.load(file)
else:
    train_gaussian_mixture(train_loader, encoder, d, batch_size, results, file_save, with_labels=True)

means_ = torch.from_numpy(means)
std_ = torch.from_numpy(std)
weights_ = torch.from_numpy(weights)

for params in encoder.parameters():
    params.requires_grad = False

for params in decoder.parameters():
    params.requires_grad = False

for params in discriminative_model.parameters():
    params.requires_grad = False


burn_in_period = 20

###Generate 500 samples from decoder
#p_y = td.Independent(td.Normal(loc=torch.zeros(10).cuda(),scale=torch.ones(10).cuda()),1)
#for i in range(50):
#    x = generate_samples_with_labels(p_z, p_y, decoder, d, L=1).cpu().data.numpy().reshape(1,1,28,28)  
#    plot_image(np.squeeze(x), os.getcwd() + "/results/generated-samples-labels/" + str(i)+ ".png" ) 
#exit()

xm_loss = np.zeros((6,10,num_epochs_test))
xm_loss_per_img = np.zeros((6,10,num_epochs_test))

xm_mse_per_img = np.zeros((6,10,num_epochs_test))
upper_bound_ = np.zeros((10,num_epochs_test))
lower_bound_ = np.zeros((10,num_epochs_test))
num_images = 10
z_kl_loss = np.zeros((6,10,num_epochs_test))
z_loglike_loss = np.zeros((6,10,num_epochs_test))


xm_mse = np.zeros((6,10,num_epochs_test))

iaf_loss = np.zeros((6,10,num_epochs_test))
iaf_mse = np.zeros((6,10,num_epochs_test))
z_loss = np.zeros((6,10,num_epochs_test))
z_mse = np.zeros((6,10,num_epochs_test))
mixture_loss = np.zeros((6,10,num_epochs_test))
mixture_mse = np.zeros((6,10,num_epochs_test))


xm_loss_NN = np.zeros((6,10,num_epochs_test))
xm_mse_NN = np.zeros((6,10,num_epochs_test))

term1 = np.zeros((6, 10, num_epochs_test))  #loglikelihood
term2 = np.zeros((6, 10, num_epochs_test))  #KL
term3 = np.zeros((6, 10, num_epochs_test))  #Entropy


K_samples_ = [1000] #100, 500, 2000

mixture_loss_samples = np.zeros((2,6,len(K_samples_)))

samples_iter = -1

for K_samples in K_samples_ :
    samples_iter += 1
    for iterations in range(1):
        for i in [-1]:
            p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
            p_y = td.Categorical(logits=torch.zeros(10).cuda())
            ### Get test loader for different missing percentage value
            #print("memory before --" )
            #print(torch.cuda.memory_allocated(device=0))
            ## MAR (0.5, 0.8)
            #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, perc_miss = i),batch_size=1)
            ## Right half missing (0)
            #test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, top_half=True, return_labels=True),batch_size=1)
            ## 4 patches of size 10*10 missing (-1)
            test_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST_Test(binarize = binary_data, patches=True, return_labels=True),batch_size=1)
            
            #print("test data loaded")
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
            xm_loglikelihood = np.zeros((num_epochs_test))

            lower_bound_all = upper_bound_all = lower_log_like_all = upper_log_like_all = lower_err = upper_err = 0

            total_loss = 0

            for data in test_loader:
                nb += 1
                if nb<20:
                    continue
                if nb==21:
                    break
                
                #print("Image : ", nb)

                ##b_data here doesn't have label concatenated but b_full does
                b_data, b_mask, b_full, labels = data
                b_data_init = b_data
                missing_counts = 1*28*28 - int(b_mask.sum())  
                b_mask = b_mask.to(device,dtype = torch.bool)
                b_full_ = b_full.reshape([1,11,28,28]).to(device,dtype = torch.float)
                
                burn_in_ = True
                random = False

                img = b_full[:,0,:,:].cpu().data.numpy().reshape([1,1,28,28])         ## added .data
                plot_image(np.squeeze(img),results + str(i) + "/images/" + str(nb%10) + "/"  +  "true.png")

                missing = b_data
                missing[~b_mask] = 0.5      
                img = missing.cpu().data.numpy() 
                plot_image(np.squeeze(img),results + str(i) + "/images/" + str(nb%10) + "/"  +  "missing.png" )

                #lower_bound =  eval_iwae_bound(iota_x = b_data.to(device,dtype = torch.float), full = b_full[:,0,:,:].reshape([1,1,28,28]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=K_samples, with_labels = True)
                upper_bound =  eval_iwae_bound(iota_x = b_full.to(device,dtype = torch.float), full = b_full[:,0,:,:].reshape([1,1,28,28]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, K=K_samples, with_labels = True, labels = labels.to(device,dtype = torch.float))
                #bound_updated_encoder = eval_iwae_bound(iota_x = b_data.to(device,dtype = torch.float), full = b_full.reshape([1,1,28,28]).to(device,dtype = torch.float), mask = b_mask,encoder = encoder_updated,decoder = decoder, p_z= p_z, d=d, K=K_samples)
                #print("Lower IWAE bound (0's) : ", lower_bound)
                print("Upper IWAE bound (true image) : ", upper_bound)
                #print("Bound with re-tuned encoder onn training dataset : ", bound_updated_encoder)

                dd = False

                if dd:
                    if random: 
                        x_logits_init = torch.zeros_like(b_data)
                        p_x_m = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=x_logits_init[~b_mask]),1)
                        b_data_init = b_data
                        b_data_init[~b_mask] = p_x_m.sample()
                        x_logits_init[~b_mask] = torch.log(b_data_init/(1-b_data_init))[~b_mask]
                    elif burn_in_:
                        x_logits_init, b_data_init, z_init = burn_in(b_data.to(device,dtype = torch.float), b_mask, labels.to(device,dtype = torch.float),  encoder, decoder, p_z, d, burn_in_period=burn_in_period, with_labels=True)
                        x_logits_init = x_logits_init.cpu()
                        b_data_init = b_data_init.cpu()
                        #b_data_init = b_data
                        #b_data_init[~b_mask] = k_neighbors_sample(b_data, b_mask) 
                        #b_data_init = b_data_init
                        #x_logits_init = torch.log(b_data_init/(1-b_data_init))

                    sampled_image = b_data_init
                    sampled_image_m = sampled_image
                    sampled_image_o = sampled_image

                
                    if iterations==0:
                        sampled_image[~b_mask] = b_full[~b_mask]    
                        sampled_image_m[~b_mask] = b_full[~b_mask]   
                        z_init =  encoder.forward(b_full.to(device,dtype = torch.float))
                        sampled_image_o = sampled_image_m
                    else:
                        #sampled_image = b_data_init
                        sampled_image_m = sampled_image
                        sampled_image_o = sampled_image

                    if not burn_in:
                        plot_image(np.squeeze(sampled_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + "init.png" )
                    else:
                        if iterations==0:
                            plot_image(np.squeeze(sampled_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + '-'  + "burn-in.png" )
                        else:
                            burn_in_image = b_data.to(device,dtype = torch.float)
                            burn_in_image[~b_mask] = torch.sigmoid(x_logits_init)[~b_mask].to(device,dtype = torch.float)
                            plot_image(np.squeeze(burn_in_image.cpu().data.numpy()),results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + '-' + "burn-in.png" )
                 
                dd = False

                if dd:
                    start_pg = datetime.now()
                    x_logits_pseudo_gibbs, x_sample_pseudo_gibbs = pseudo_gibbs(sampled_image.to(device,dtype = torch.float), b_data.to(device,dtype = torch.float), b_mask, encoder, decoder, p_z, d, results, iterations, T=num_epochs_test, nb=nb, K = 1)
                    end_pg = datetime.now()
                    diff_z = end_pg - start_pg
                    print(" Time taken for pseudo-gibbs", diff_z.total_seconds())

                    #Impute image with pseudo-gibbs
                    pseudo_gibbs_image = b_data.to(device,dtype = torch.float)
                    pseudo_gibbs_image[~b_mask] = torch.sigmoid(x_logits_pseudo_gibbs[~b_mask])
                    plot_image(np.squeeze(pseudo_gibbs_image.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + '-' + "pseudo-gibbs.png" )

                    ##M-with-gibbs sampler
                    start_m = datetime.now()
                    m_nelbo, m_error, x_full_logits, m_loglike =  m_g_sampler(iota_x = sampled_image_m.to(device,dtype = torch.float), full = b_full.to(device,dtype = torch.float), mask = b_mask,encoder = encoder,decoder = decoder, p_z= p_z, d=d, results = results, nb = nb, iterations = iterations, K= 1, T=num_epochs_test)
                    end_m = datetime.now()
                    diff_z = end_m - start_m
                    print(" Time taken for metropolis within gibbs", diff_z.total_seconds())
                
                    m_loss += np.array(m_nelbo).reshape((num_epochs_test))
                    m_loglikelihood += np.array(m_loglike).reshape((num_epochs_test))
                    m_elbo -= np.array(m_nelbo).reshape((num_epochs_test))
                    m_mse += np.array(m_error).reshape((num_epochs_test))

                    #Impute image with metropolis-within-pseudo-gibbs
                    metropolis_image = b_data.to(device,dtype = torch.float)
                    metropolis_image[~b_mask] = torch.sigmoid(x_full_logits[~b_mask])
                    plot_image(np.squeeze(metropolis_image.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/" + str(iterations) + '-' +"metropolis-within-pseudo-gibbs.png" )

                    if iterations==-1:
                        xm_params = torch.log(b_full/(1-b_full))[~b_mask]
                    else:
                        xm_params = x_logits_init[~b_mask]

                    ##Optimize q(x_m) with burn-in
                    start_o = datetime.now()
                    xm_nelbo_, xm_error_ = optimize_q_xm(num_epochs = num_epochs_test, xm_params = xm_params, z_params = z_init, b_data = b_data.to(device,dtype = torch.float), sampled_image_o = sampled_image_o.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), b_full = b_full.to(device,dtype = torch.float), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb= nb, file = "q_xm-all.png", p_z = p_z, K_samples = K_samples  )
                    end_o = datetime.now()
                    diff_o = end_o - start_o
                    print(" Time taken for optimizing q(x_m)", diff_o.total_seconds())
                
                    xm_loss[iterations, nb%10, : ] = xm_nelbo_
                    xm_mse[iterations, nb%10, : ] = xm_error_

                    ##Optimize q(x_m) with nearest-neighbour 
                    b_data_init_ = b_data
                    b_data_init_ = b_data_init_.to(device,dtype = torch.float)
                    b_data_init_[~b_mask] = k_neighbors_sample(b_data, b_mask) 

                    x_logits_init = torch.log(b_data_init_/(1-b_data_init_))

                    plot_image(np.squeeze(b_data_init_.cpu().data.numpy()), results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + '-' + "NN-init.png" )
                    prefix = results + str(i) + "/images/" + str(nb%10) + "/"  + str(iterations) + '-'
                    plot_images_in_row(num_epochs = num_epochs_test, loc1 = results + str(i) + "/images/" + str(nb%10) + "/"  +  "true.png" , loc2 = results + str(i) + "/images/" + str(nb%10) + "/"  +  "missing.png" , loc3 = prefix + "burn-in.png" , loc4 = prefix + "NN-init.png", loc5 = prefix + "NN-init.png" , file = prefix + "image-inits.png")

                    if iterations==-1:
                        xm_params = torch.log(b_full/(1-b_full))[~b_mask]
                    else:
                        xm_params = x_logits_init[~b_mask]

                    gc.collect()
                    start_o = datetime.now()
                    xm_nelbo_, xm_error_ = optimize_q_xm(num_epochs = num_epochs_test, xm_params = xm_params, z_params = z_init, b_data = b_data_init_.to(device,dtype = torch.float), sampled_image_o = sampled_image_o.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), b_full = b_full.to(device,dtype = torch.float), encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb = nb, file = "q_xm-all-NN.png" , p_z = p_z, K_samples = K_samples  )
                    end_o = datetime.now()
                    diff_o = end_o - start_o
                    #print(" Time taken for optimizing q(x_m)", diff_o.total_seconds())

                    xm_loss_NN[iterations, nb%10, : ] = xm_nelbo_
                    xm_mse_NN[iterations, nb%10, : ] = xm_error_

                    b_data[~b_mask] = 0
                    if iterations==-1:
                        z_init =  encoder.forward(b_full.to(device,dtype = torch.float))
                    else:
                        z_init =  encoder.forward(b_data.to(device,dtype = torch.float))

                    start_iaf = datetime.now()
                    xm_nelbo_, xm_error_ = optimize_IAF(num_epochs = num_epochs_test, xm_params = xm_params, z_params = z_init, b_data = b_data.to(device,dtype = torch.float), sampled_image_o = sampled_image_o.to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), b_full = b_full.to(device,dtype = torch.float), p_z = p_z, encoder = encoder, decoder = decoder, device = device, d = d, results = results, iterations = iterations, nb=nb, K_samples = K_samples )
                    end_iaf = datetime.now()
                    diff_iaf = end_iaf - start_iaf
                    print("Time taken for optimizing IAF : ", diff_iaf.total_seconds())

                    iaf_loss[iterations, nb%10, : ] = xm_nelbo_
                    iaf_mse[iterations, nb%10, : ] = xm_error_

                    b_data[~b_mask] = 0

                    if iterations==-1:
                        z_init =  encoder.forward(b_full.to(device,dtype = torch.float))
                    else:
                        z_init =  encoder.forward(b_data.to(device,dtype = torch.float))

                ##From here
                b_data[~b_mask] = 0
                start_z = datetime.now()
                z_nelbo_, z_error_ = optimize_z_with_labels(num_epochs = num_epochs_test, p_z = p_z, p_y = p_y, b_data = b_data.to(device,dtype = torch.float), b_full = b_full[:,0,:,:].reshape([1,1,28,28]).to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool),  encoder = encoder, decoder = decoder, discriminative_model = discriminative_model, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples)
                end_z = datetime.now()
                diff_z = end_z - start_z
                print("Time taken for optimizing z : ", diff_z.total_seconds())

                z_loss[iterations, nb%10, : ] = z_nelbo_
                z_mse[iterations, nb%10, : ] = z_error_

                prefix = results + str(i) + "/images/" +  str(nb%10) + "/"  + str(iterations) + '-' 

                #print(means, std, weights)
                means_ = torch.from_numpy(means)
                std_ = torch.from_numpy(std)
                weights_ = torch.from_numpy(weights)

                b_data[~b_mask] = 0
                #p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(probs=weights_.cuda()), td.Independent(td.Normal(means_.cuda(), std_.cuda()), 1))
                start_mix = datetime.now()
                z_nelbo_, z_error_ = optimize_mixture_labels(num_epochs = num_epochs_test, means_pz= means_, std_pz = std_, weights_pz = weights_, p_y = p_y,  b_data = b_data.to(device,dtype = torch.float), b_full = b_full[:,0,:,:].reshape([1,1,28,28]).to(device,dtype = torch.float), b_mask = b_mask.to(device,dtype = torch.bool), encoder = encoder, decoder = decoder, discriminative_model = discriminative_model, device = device, d = d, results = results, iterations = iterations, nb=nb , K_samples = K_samples,  labels = labels.to(device,dtype = torch.float))
                end_mix = datetime.now()
                diff_mix = end_mix - start_mix
                print("Time taken for optimizing mixtures : ", diff_mix.total_seconds())


                mixture_loss[iterations, nb%10, : ] = z_nelbo_
                mixture_mse[iterations, nb%10, : ] = z_error_

                mixture_loss_samples[nb%10, iterations, samples_iter] = z_nelbo_[-1]

                #plot_all_images(i, nb, iterations, results + str(i) + "/compiled/" + str(nb%10)  + str(iterations)  + 'image-comparison.png', prefix)
                gc.collect()
                
plot_loss_vs_sample_size(mixture_loss_samples, K_samples_, results + str(i) + "/compiled/" )
plot_5runs(num_epochs_test, xm_loss, iaf_loss, z_loss, xm_loss_NN, mixture_loss,results, -1)

for image in range(10):
    #print(image)
    value = mixture_loss[0,image,-1]
    compare_ELBO(num_epochs_test, xm_loss[:,image,:], iaf_loss[:,image,:], z_loss[:,image,:], xm_loss_NN[:,image,:], mixture_loss[:,image,:] , results, -1, ylim1= -1600,ylim2 = -1400,  image = image) #



