import torch
import numpy as np
import torch.distributions as td
import torch.nn.functional as F
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
import pyro 
from mixture import *
from gaussian import *
import os
import pickle
from plot import *
import matplotlib.pyplot as plt


def mean(x):
    if x != torch.tensor(0.5):
        return x/(2*x-1) + 1/(2*torch.atan(1-2*x))
    else:
        return torch.tensor(0.5)

def log_C(x):
    if x != torch.tensor(0.5):
        #print(x, 2/(1-2*x))
        #print(torch.atan(1-2*x))
        return (2/(1-2*x))*torch.atan(1-2*x)
    else:
        return torch.tensor(2)

def entropy(x,device):
    if torch.is_nonzero(x) : 
        return 1 -  x/(1-torch.exp(-x)) -torch.log(x/(torch.exp(x)-1)) 
    else :
        return torch.tensor(0).to(device,dtype = torch.float)

def plot_image(img, file='true.png', missing_pattern_x = None, missing_pattern_y = None):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    if missing_pattern_x is not None: 
        plt.scatter(missing_pattern_y, missing_pattern_x)

    plt.show()
    plt.savefig(file)
    plt.close()


def plot_labels_in_row(images, logqy,  file, data='mnist'):

    fig = plt.figure(figsize=(4, 1))

    # setting values to rows and column variables
    rows = 1
    columns = 10
    probs_qy = logqy.cpu().data.numpy()

    for i in range(10):
        fig.add_subplot(rows, columns, i+1)
        # showing image
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        #plt.imshow(image1)
        plt.axis('off')
        plt.title(str(np.round(probs_qy[0,i], decimals=3)), fontsize = 7)

    plt.show()
    plt.savefig(file)
    plt.close()


def mvae_loss_simple(iota_x, mask, encoder, decoder,p_z, d, K=1):
    batch_size = iota_x.shape[0]
    #p = iota_x.shape[2]
    #q = iota_x.shape[3]
    p=28
    q=28
    iota_x = torch.reshape(iota_x, (batch_size, p*q))
    #mask = torch.reshape(mask, (batch_size, p*q))

    #print(iota_x.size())
    out_encoder = encoder.forward(iota_x)

    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([K])

    zgivenx_flat = zgivenx.reshape([K*batch_size,d])
    #print(zgivenx.size(),zgivenx_flat.size(), iota_x.size())

    all_logits_obs_model = decoder.forward(zgivenx_flat)

    mask_ = mask.reshape([batch_size,p*q])
    iota_x_ = iota_x.reshape([batch_size,p*q])
    #print(iota_x.size(), mask.size())
    data_flat = torch.Tensor.repeat(iota_x_,[K,1]).reshape([-1,1]) # iota_x.reshape([-1,1])
    tiledmask = torch.Tensor.repeat(mask_,[K,1])                   # mask.reshape([batch_size,28*28])

    #print(data_flat.size(),tiledmask.size())
    #print(data_flat)
    
    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    #all_log_pxgivenz_flat = td.bernoulli.Bernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p*q])
    #print(all_log_pxgivenz_flat.size(), all_log_pxgivenz.size())

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,[1]).reshape([K,batch_size])
    logpxobsgivenz_ = torch.sum(all_log_pxgivenz,[1]).reshape([K,batch_size])

    #logpxobsgivenz = torch.sum(all_log_pxgivenz,[1]).reshape([K,batch_size])

    logpz = p_z.log_prob(zgivenx)
    
    logq = q_zgivenxobs.log_prob(zgivenx)

    #print(logpxobsgivenz.size(), logpz.size(), logq.size())
    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
    #log_like = torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0))

    log_like = torch.mean(torch.sum(logpz + logpxobsgivenz_,0))
    #print(torch.sum(logpz), torch.sum(logpxobsgivenz_))
    return neg_bound, log_like

def mvae_loss(iota_x, mask, encoder, decoder, p_z, d, K=1, with_labels=False, labels=None):
    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]

    out_encoder = encoder.forward(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([K])
    if with_labels:
        labels = labels.reshape(1,labels.shape[0],10)
        #labels = labels.repeat(K)
        zgivenx_y = torch.cat((zgivenx,labels),2)
        zgivenx_flat = zgivenx_y.reshape([K*batch_size,d+10])
    else:
        zgivenx_flat = zgivenx.reshape([K*batch_size,d])

    all_logits_obs_model = decoder.forward(zgivenx_flat)
    #print(all_logits_obs_model.size())

    channels = 1
    mask_ = mask.reshape([batch_size,channels*p*q])
    iota_x_ = iota_x[:,0,:,:].reshape([batch_size,channels*p*q])
    data_flat = torch.Tensor.repeat(iota_x_,[K,1]).reshape([-1,1]) # iota_x.reshape([-1,1])
    tiledmask = torch.Tensor.repeat(mask_,[K,1])                   # mask.reshape([batch_size,28*28])

    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,[1]).reshape([K,batch_size])
    logpxobsgivenz_ = torch.sum(all_log_pxgivenz,[1]).reshape([K,batch_size])

    logpz = p_z.log_prob(zgivenx)
    #q_zgivenxobs is q(z|x,y) if with_labels=True
    logq = q_zgivenxobs.log_prob(zgivenx)

    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
    #log_like = torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0))
    log_like = torch.mean(torch.sum(logpz + logpxobsgivenz_,0))
    #print(torch.sum(logpz), torch.sum(logpxobsgivenz_))
    return neg_bound, log_like

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def mvae_loss_svhn(iota_x, mask, encoder, decoder, p_z, d, K=1):
    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]

    #print(iota_x.size())
    out_encoder = encoder.forward(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K*batch_size,d])
    #print(zgivenx.size(),zgivenx_flat.size(), iota_x.size())

    all_logits_obs_model = decoder.forward(zgivenx_flat)

    #print(all_logits_obs_model.size())

    mask_ = mask.reshape([batch_size,channels*p*q])
    iota_x_ = iota_x.reshape([batch_size,channels*p*q])
    data_flat = torch.Tensor.repeat(iota_x_,[K,1]).reshape([-1,1]) # iota_x.reshape([-1,1])
    tiledmask = torch.Tensor.repeat(mask_,[K,1])                   # mask.reshape([batch_size,28*28])

    sigma_decoder = decoder.get_parameter("log_sigma")
    sigma_decoder = softclip(sigma_decoder, -6)

    #rec = gaussian_nll(x_hat, log_sigma, x).sum()
    ##Do the sigma normal distribution....
    all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale =  sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,[1]).reshape([K,batch_size])
    logpxobsgivenz_ = torch.sum(all_log_pxgivenz,[1]).reshape([K,batch_size])

    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)
    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
    #log_like = torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0))
    log_like = torch.mean(torch.sum(logpz + logpxobsgivenz_,0))
    #print(torch.sum(logpz), torch.sum(logpxobsgivenz_))
    return neg_bound, log_like

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    #print("inside kl func --- ")
    #print(mean_sq.shape, stddev_sq.shape)
    #print(torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1,1))
    return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1,1)

def xm_loss_noise(iota_x, mask, p_z, xm_params, encoder, decoder , device, d, beta, K=1, K_z=1, std=None):
    batch_size = iota_x.shape[0]
    p = iota_x.shape[2]
    q = iota_x.shape[3]
    mp = len(iota_x[~mask])

    p_xm = td.continuous_bernoulli.ContinuousBernoulli(logits=xm_params.reshape([-1,1]))

    mse_loss = torch.nn.MSELoss(reduction="mean")
    #logp_ygivenx = -(p*q*torch.log(std*torch.sqrt(2*np.pi)) + (1/2*std*std)*mse_loss(iota_x.reshape(-1),expected_x))

    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1])       #.reshape([-1,1])       
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])         #.reshape([-1,1]) 

    expected_x = p_xm.rsample([K]).reshape(-1)
    a = iota_x_[~mask_]
    b = expected_x[~mask_.reshape(-1)]
    #print(iota_x_.shape, mask_.shape)
    #if train:
    #    iota_x_[~mask_] =  p_xm.rsample([K]).reshape(-1)            ## plot image and check here...
    #print(p_xm.rsample([K]).shape, iota_x_.shape)

    ###calculating the loss term for the noisy image with std =std
    #expected_x = torch.Tensor.repeat(torch.sigmoid(xm_params),[K,1,1,1]) 

    #logp_ygivenx = - (1/(2*std*std))*mse_loss(iota_x_.reshape(-1),expected_x)
    #logp_ygivenx = -(p*q*torch.log(std*torch.sqrt(torch.tensor(2*torch.pi))) + (1/(2*std*std))*mse_loss(iota_x_.reshape(-1),expected_x))
    logp_ygivenx = -(1/(2*std*std))*mse_loss(a.reshape(-1),b)*mp ##normal noise

    #print(logp_ygivenx)

    expected_x = expected_x.reshape([K,1,p,q])
    out_encoder = encoder.forward(expected_x) 
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
    zgivenx = q_zgivenxobs.rsample([K_z])
    #print(zgivenx.shape)
    zgivenx_flat = zgivenx.reshape([K*K_z*batch_size,d])
    #print(zgivenx_flat.shape)                           # sample one z
    all_logits_obs_model = decoder.forward(zgivenx_flat)
    #print(all_logits_obs_model)
    #data_flat = torch.Tensor.repeat(data,[K,1]).reshape([-1,1])
    #data_flat = iota_x_.reshape([-1,1])
    data_flat = expected_x.reshape([-1,1])
    #tiledmask = torch.Tensor.repeat(mask_,[K,1])        #reshape([batch_size,28*28])
    tiledmask = mask_.reshape([K*batch_size,28*28]).cuda()

    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    #print(all_log_pxgivenz_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p*q])

    logpxobsgivenz = torch.sum(all_log_pxgivenz,1).reshape([K,batch_size])          ####*tiledmask??????   *tiledmask
    #logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size]) 
    #logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 

    logpxobsgivenz_ = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])            ##

    logpz = p_z.log_prob(zgivenx).reshape([K,batch_size])
    ##print its shape to see what shape data[~mask] should be in
    logq = q_zgivenxobs.log_prob(zgivenx).reshape([K,batch_size])
    #print(iota_x_[~mask_].reshape(K*batch_size,mp))

    k_l1 = torch.distributions.kl.kl_divergence(q_zgivenxobs, p_z)
    #print(k_l1)
    k_l = latent_loss(out_encoder[...,:d], torch.nn.Softplus()(out_encoder[...,d:]))
    #print(k_l.shape)
    #print(xm_params.size(), iota_x_[~mask_].shape)

    #miss_log_px_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=torch.Tensor.repeat(xm_params,[K,1,1,1]).reshape([-1,1])).log_prob(iota_x_.reshape(-1,1))
    #miss_log_px = miss_log_px_flat.reshape([K*batch_size,28*28])
    #logqm = torch.sum(miss_log_px,1).reshape([K,batch_size])    

    #q_entropy = torch.sum(td.continuous_bernoulli.ContinuousBernoulli(logits=xm_params.reshape([-1,1])).entropy())
    q_entropy = torch.tensor(0).to(device,dtype = torch.float)

    for x in xm_params: 
        q_entropy += entropy(x,device) 

    #logqm = p_xm.log_prob(iota_x_[~mask_].reshape(K*batch_size,mp)) 
    #print(logpxobsgivenz.shape, logpz.shape, logq.shape, logqm.shape)
    #print(q_entropy)
    #print(logpxobsgivenz.item(),logpz.item(),logq.item(), logqm.item(), q_entropy.item())
    #print(torch.logsumexp(logpxobsgivenz + logpz - logq - logqm,0))
    #neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq + q_entropy,0))
    #print(k_l.shape, logpxobsgivenz.shape, q_entropy.shape)
    #alpha-annelaing
    neg_bound = - (torch.mean(logpxobsgivenz  - k_l + logp_ygivenx)  + beta*q_entropy) ##(1/beta)*logpxmissgivenz 
    #beta-annealing
    #neg_bound = - (torch.mean(logpxobsgivenz + (beta)*logpxmissgivenz - k_l) + (beta)* q_entropy)

    #neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq - logqm,0))
    #print(all_logits_obs_model.shape)
    #print("All components of loss ---")
    #print(torch.sum(logpxobsgivenz), torch.sum(logpz), torch.sum(logq), torch.prod(torch.sigmoid(logqm)))
    #print("loss :", neg_bound.item())
    #print(logqm,torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0)))
    #log_like = torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0))
    #approx_bound = -torch.mean(torch.logsumexp(logpxobsgivenz_ + logpz - logq,0))
    approx_bound = torch.mean(logpz + logpxobsgivenz + logp_ygivenx)
    #print(torch.mean(logpz), torch.mean(logpxobsgivenz_))
    return neg_bound, approx_bound, -(torch.mean(logpxobsgivenz)), torch.mean(k_l), -torch.mean(q_entropy)

def inverse_autoregressive_flow(q_zgivenxobs, p_z, autoregressive_nn, d, K):
    #q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
    #z_ =  q_zgivenxobs.rsample()
    T= 1
    batch_size = 1

    for i in range(T):
        transform = AffineAutoregressive(autoregressive_nn).cuda()
        pyro.module("my_transform", transform)  
        flow_dist = pyro.distributions.torch.TransformedDistribution(q_zgivenxobs, [transform])
        z = flow_dist.sample([K])  
        #print(z.shape)
        logqz = flow_dist.log_prob(z).reshape(K,batch_size)
        logpz = p_z.log_prob(z).reshape(K,batch_size)
        kl = logqz - logpz
        #print(flow_dist.log_prob(z).shape)
    return z, kl, flow_dist

def z_loss_(iota_x, mask, p_z, z_params, encoder, decoder, device, d, K=1, K_z=1, data='mnist', iaf=False, autoregressive_nn=None, autoregressive_nn2 = None, autoregressive_nn3 = None, autoregressive_nn4 = None, autoregressive_nn5 = None, evaluate=False, full = None):
    ##iota_x is the missing image, sampled is not to be used, ...
    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1]) 
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])  

    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]
    mp = len(iota_x[~mask])
    
    if data=='svhn':
        sigma_decoder = decoder.get_parameter("log_sigma")
    #iota_x_[~mask_] =  p_xm.rsample([K]).reshape(-1)   
    q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d], scale=torch.nn.Softplus()(z_params[...,d:])),1)

    k_l = torch.zeros(K,batch_size)
    zgivenx = torch.zeros(K*batch_size,d)
    
    if iaf:
        #zgivenx, k_l, flow_dist = inverse_autoregressive_flow(q_zgivenxobs, p_z, autoregressive_nn, d, K)
        T=1
        for i in range(T):
            transform = AffineAutoregressive(autoregressive_nn).cuda()
            transform2 = AffineAutoregressive(autoregressive_nn2).cuda()
            #transform3 = AffineAutoregressive(autoregressive_nn3).cuda()
            #transform4 = AffineAutoregressive(autoregressive_nn4).cuda()
            #transform5 = AffineAutoregressive(autoregressive_nn5).cuda()

            pyro.module("my_transform", transform)  
            flow_dist = pyro.distributions.torch.TransformedDistribution(q_zgivenxobs, [transform, transform2]) #, transform2, transform3, transform4, transform5
            zgivenx = flow_dist.rsample([K])  
            #print(z.shape)
            logqz = flow_dist.log_prob(zgivenx).reshape(K,batch_size)
            logpz = p_z.log_prob(zgivenx).reshape(K,batch_size)    
            k_l = torch.mean(logqz - logpz)
    else:
        zgivenx = q_zgivenxobs.rsample([K*K_z])
        k_l = latent_loss(z_params[...,:d], torch.nn.Softplus()(z_params[...,d:]))
        logqz = q_zgivenxobs.log_prob(zgivenx).reshape(K,batch_size)
        logpz = p_z.log_prob(zgivenx).reshape(K,batch_size)    #print("sampled z ---",zgivenx)

    #print("KL from sampled z ---", k_l)
    zgivenx_flat = zgivenx.reshape([K*K_z*batch_size,d])
    all_logits_obs_model = decoder.forward(zgivenx_flat)
    #print(all_logits_obs_model, torch.sum(torch.isnan(all_logits_obs_model)))

    data_flat = iota_x_.reshape([-1,1])

    if evaluate:
        full_ = torch.Tensor.repeat(full,[K,1,1,1]) 
        data_flat = full_.reshape([-1,1])

    tiledmask = mask_.reshape([K*batch_size,channels*p*q]).cuda()

    if data=='mnist':
        all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    else:
        all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale = sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
        #print(all_log_pxgivenz_flat)

    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size]) 
    logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 
    #print(torch.sum(torch.isnan(logpxobsgivenz)))

    if not iaf:
        neg_bound = - (torch.mean(logpxobsgivenz)  - k_l) ## remove logxmissgivenz
    else:
        neg_bound = - (torch.mean(logpxobsgivenz) - torch.mean(logqz - logpz))

    if evaluate:
        neg_bound = - torch.logsumexp(logpxmissgivenz + logpz - logqz, 0)
    #print(-torch.mean(logpxobsgivenz).item(), torch.mean(k_l).item())
    #print(logpxobsgivenz, k_l)
    #print(neg_bound)
    approx_bound = torch.mean(logpz + logpxobsgivenz)

    if iaf:
        return neg_bound, approx_bound, -torch.mean(logpxobsgivenz), torch.mean(logqz - logpz), flow_dist
    else:
        return neg_bound, approx_bound, -torch.mean(logpxobsgivenz), torch.mean(k_l), z_params


def z_loss_with_labels(iota_x, mask, p_z, p_y, means, scales, logits_y, encoder, decoder, device, d, K, K_z, data='mnist', iwae='False', evaluate=False, full = None,iaf=False, autoregressive_nn = None, autoregressive_nn2= None):
    
    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]

    q_y = td.Categorical(logits=logits_y)
    probs_qy = torch.nn.functional.softmax(logits_y, dim=1)
    probs_py = 0.1 + torch.zeros(10).cuda()
    kl_y = torch.sum((torch.log(probs_qy) - torch.log(probs_py))*probs_qy)
    ELBO = -kl_y
    #print(kl_y, ELBO)

    K = int(K/10)
    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1]) 
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])  
    tiledmask = mask_.reshape([K*batch_size,channels*p*q]).cuda()
    data_flat = iota_x_.reshape([-1,1])

    if evaluate:
        full_ = torch.Tensor.repeat(full,[K,1,1,1]) 
        data_flat = full_.reshape([-1,1])
        ELBO = 0
        #file_save = os.getcwd() + "/results/mnist-False--1/classification-gmms-per-class.pkl"
        #with open(file_save, 'rb') as file:
        #    means_pz,std_pz,weights_pz = pickle.load(file)

        #means_pz = torch.from_numpy(means_pz)
        #std_pz = torch.from_numpy(std_pz)
        #weights_pz = torch.from_numpy(weights_pz)
        images = np.zeros((10,28,28))

    for i in range(10):
        label = torch.zeros(K*K_z, 10)
        label[:, i] = 1.0

        if evaluate:
            logpy = torch.Tensor.repeat(torch.log(probs_py[i]), [K]).reshape([K,batch_size])
            logqy = torch.Tensor.repeat(torch.log(probs_qy[0,i]), [K]).reshape([K,batch_size])

        q_zgiveny = td.Independent(td.Normal(loc=means[i,:], scale=torch.nn.Softplus()(scales[i,:])),1)

        if iaf:
            transform = AffineAutoregressive(autoregressive_nn).cuda()
            transform2 = AffineAutoregressive(autoregressive_nn2).cuda()
            #transform = AffineAutoregressive(autoregressive_nn[i]).cuda()
            #transform2 = AffineAutoregressive(autoregressive_nn2[i]).cuda()
            pyro.module("my_transform", transform)  
            flow_dist = pyro.distributions.torch.TransformedDistribution(q_zgiveny, [transform, transform2]) #, transform2, transform3, transform4, transform5
            zgiveny = flow_dist.rsample([K])  
            logqz = flow_dist.log_prob(zgiveny).reshape(K,batch_size)
            logpz = p_z.log_prob(zgiveny).reshape(K,batch_size)    
            k_l_z = torch.mean(logqz - logpz)
        else:
            zgiveny = q_zgiveny.rsample([K*K_z])
            k_l_z = latent_loss(means[i,:].reshape(1,d), torch.nn.Softplus()(scales[i,:]).reshape(1,d))
            logqz = q_zgiveny.log_prob(zgiveny).reshape(K,batch_size)
            logpz = p_z.log_prob(zgiveny).reshape(K,batch_size)    #print("sampled z ---",zgivenx)
        
        zgivenx_y = torch.cat((zgiveny,label.to(device,dtype = torch.float)),1)
        zgivenxy_flat = zgivenx_y.reshape([K*K_z*batch_size,d+10])
        all_logits_obs_model = decoder.forward(zgivenxy_flat)

        #print(torch.isnan(all_logits_obs_model).any())
        if data=='mnist':
            all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
        else:
            all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale = sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
            #print(all_log_pxgivenz_flat)

        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])
        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size]) 
        logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 

        if evaluate:
            #print(logpxobsgivenz.shape, logpxmissgivenz.shape, logpz.shape, logpy.shape, logqy.shape, logqz.shape, probs_qy[0,i])
            ELBO += torch.logsumexp(logpxmissgivenz + logpz + logpy - logqy - logqz,0)*probs_qy[0,i]
            index_ = torch.argmax(logpxmissgivenz + logpz + logpy - logqy - logqz)
            print(index_)
            predicted_image = iota_x
            predicted_image[~mask] = torch.sigmoid(all_logits_obs_model.reshape(K,1,1,28,28)[index_,~mask])
            print(logpxmissgivenz[index_] + logpz[index_] + logpy[index_] - logqy[index_] - logqz[index_])
            images[i] = np.squeeze(predicted_image.cpu().data.numpy())
            
            #plot_image(np.squeeze(img),)
        else:
            if not iaf:
                ELBO += probs_qy[0,i]*(torch.mean(logpxobsgivenz) - k_l_z)[0]
            else:
                ELBO += probs_qy[0,i]*(torch.mean(logpxobsgivenz) - torch.mean(logqz - logpz))

    if evaluate:
        results = os.getcwd() + "/results/mnist-False--1/compiled/"
        if not iaf:
            plot_labels_in_row(images, probs_qy, results + "-gaussian_labels.png")
        else:
            plot_labels_in_row(images, probs_qy, results + "-iaf_labels.png")

    neg_bound = - ELBO
    approx_bound = ELBO

    return neg_bound, approx_bound, torch.mean(logpxobsgivenz), torch.mean(kl_y), torch.mean(k_l_z)

def mixture_loss_labels(iota_x, mask, p_z,  means, scales, logits, logits_y, decoder, device, d, K=1, K_z=1, data='mnist', beta=1, num_components=10, iwae=False, evaluate=False, full = None):
    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]

    mix = td.categorical.Categorical(torch.ones(d,).cuda())  
    p_y = td.Categorical(logits=torch.zeros(1, num_components))
    q_y = td.Categorical(logits=logits_y)

    probs_qy = torch.nn.functional.softmax(logits_y, dim=1)
    probs_py = 0.1 + torch.zeros(10).cuda()
    kl_y = torch.sum((torch.log(probs_qy) - torch.log(probs_py))*probs_qy)
    ELBO = -kl_y

    K = int(K/10)
    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1]) 
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])  
    tiledmask = mask_.reshape([K*batch_size,channels*p*q]).cuda()
    data_flat = iota_x_.reshape([-1,1])

    num_components = logits.shape[1]
   
    if data=='svhn':
        sigma_decoder = decoder.get_parameter("log_sigma")

    neg_bound = 0
    approx_bound = 0

    if evaluate:
        full_ = torch.Tensor.repeat(full,[K,1,1,1]) 
        data_flat = full_.reshape([-1,1])
        ELBO = 0
        images = np.zeros((10,28,28))
    if iwae:
        ELBO = 0


    for comp in range(10):
        label = torch.zeros(K*K_z,10)
        label[:, comp] = 1.0
        ##Calculating log probabilities of labels
        logpy = torch.Tensor.repeat(torch.log(probs_py[comp]), [K]).reshape([K,batch_size]) 
        logqy = torch.Tensor.repeat(torch.log(probs_qy[0,comp]), [K]).reshape([K,batch_size]) 

        #p_z = td.mixture_same_family.MixtureSameFamily(td.Categorical(probs=weights_pz[comp].cuda()), td.Independent(td.Normal(means_pz[comp].cuda(), std_pz[comp].cuda()), 1))
        #print("Shape of py --", logpy.shape)
        p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

        q_zgiveny = ReparameterizedNormalMixture1d(logits[comp].reshape(1,num_components), means[comp].reshape(1, num_components, d), torch.nn.Softplus()(scales[comp]).reshape(1, num_components, d))
        zgiveny = q_zgiveny.rsample([K*K_z]).reshape(K*K_z,d) 

        ##Calculating log probabilities of z samples
        logpz = p_z.log_prob(zgiveny).reshape([K,batch_size])
        logqz = q_zgiveny.log_prob(zgiveny).reshape([K,batch_size])
        #print("Shape of pz --", logpz.shape)

        k_l_z = torch.mean(logqz - logpz)
        #print(zgiveny.shape)
        zgivenx_y = torch.cat((zgiveny,label.to(device,dtype = torch.float)),1)
        zgivenxy_flat = zgivenx_y.reshape([K*K_z*batch_size,d+10])
        all_logits_obs_model = decoder.forward(zgivenxy_flat)

        if data=='mnist':
            all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
        else:
            all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale = sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)

        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])
        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size]) 
        logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 
        #print("Shape of observed prob --", logpxobsgivenz.shape)

        if evaluate:
            ELBO += torch.logsumexp(logpxmissgivenz + logpz + logpy - logqy - logqz,0)*probs_qy[0,comp]
            index_ = torch.argmax(logpxmissgivenz + logpz + logpy - logqy - logqz)
            predicted_image = iota_x
            predicted_image[~mask] = torch.sigmoid(all_logits_obs_model.reshape(K,1,1,28,28)[index_,~mask])
            print(logpxmissgivenz[index_] + logpz[index_] + logpy[index_] - logqy[index_] - logqz[index_])
            #img = predicted_image.cpu().data.numpy()
            images[comp] = np.squeeze(predicted_image.cpu().data.numpy())
        elif iwae:
            #print(torch.logsumexp(logpxobsgivenz + logpz + logpy - logqy - logqz,0).shape)
            ELBO += torch.logsumexp(logpxobsgivenz + logpz + logpy - logqy - logqz,0)*probs_qy[0,comp]
        else:
            ELBO += probs_qy[0,comp]*(torch.mean(logpxobsgivenz) - k_l_z)
            

        #neg_bound += (logpxobsgivenz + beta*logpz + beta*logpy - beta*logqz - beta*logqy)*torch.exp(logqy) 
        approx_bound += logpz + logpxobsgivenz
        
    if evaluate:
        results = os.getcwd() + "/results/mnist-False--1/compiled/"
        plot_labels_in_row(images, probs_qy, results + "-mixture_labels.png")

    neg_bound = - (ELBO).to(device,dtype = torch.float) # + beta*td.Categorical(logits=logits).entropy() - beta*td.Categorical(logits=logits).entropy().detach() 
    approx_bound = torch.mean(approx_bound)

    return neg_bound, approx_bound, -torch.mean(logpxobsgivenz), torch.mean(kl_y)


def z_loss_with_labels_iwae(iota_x, mask, p_z, p_y, means, scales, logits_y, encoder, decoder, device, d, K, K_z, data='mnist'):

    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]

    q_y = td.Categorical(logits=logits_y)
    probs_qy = torch.nn.functional.softmax(logits_y, dim=1)
    probs_py = 0.1 + torch.zeros(10).cuda()

    K = int(K/10)

    logpy = torch.zeros((K,1)).to(device,dtype = torch.float)
    logqy = torch.zeros((K,1)).to(device,dtype = torch.float)
    logpz = torch.zeros((K,1)).to(device,dtype = torch.float)
    logqz = torch.zeros((K,1)).to(device,dtype = torch.float)
    logpxobsgivenz = torch.zeros((K,1)).to(device,dtype = torch.float)

    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1]) 
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])  
    tiledmask = mask_.reshape([K*batch_size,channels*p*q]).cuda()
    data_flat = iota_x_.reshape([-1,1])

    for i in range(10):
        #print(i)
        label = torch.zeros(K*K_z, 10)
        label[:, i] = 1.0
        #print(torch.log(probs_py[i]),torch.log(probs_qy[0,i])) 
        logpy = torch.Tensor.repeat(torch.log(probs_py[i]), [K])
        logqy = torch.Tensor.repeat(torch.log(probs_qy[0,i]), [K])
        #print(logpy.shape)

        q_zgiveny = td.Independent(td.Normal(loc=means[i,:], scale=torch.nn.Softplus()(scales[i,:])),1)
        zgiveny = q_zgiveny.rsample([K*K_z]) # 
        #print(zgiveny.shape, p_z.log_prob(zgiveny).shape)

        logpz = p_z.log_prob(zgiveny)
        logqz = q_zgiveny.log_prob(zgiveny)

        zgivenx_y = torch.cat((zgiveny,label.to(device,dtype = torch.float)),1)
        zgivenxy_flat = zgivenx_y.reshape([K*K_z*batch_size,d+10])
        all_logits_obs_model = decoder.forward(zgivenxy_flat)

        #print(torch.isnan(all_logits_obs_model).any())
        if data=='mnist':
            all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
        else:
            all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale = sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)

        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])
        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1)
        logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 

        if i==0:
            ELBO = torch.logsumexp(logpxobsgivenz + logpz + logpy - logqy - logqz,0)*probs_qy[0,i]
        else:
            ELBO += torch.logsumexp(logpxobsgivenz + logpz + logpy - logqy - logqz,0)*probs_qy[0,i]
        #print(logpxobsgivenz.shape)

    print(ELBO)
    #print(logpxobsgivenz + logpz + logpy - logqy - logqz, (logpxobsgivenz + logpz + logpy - logqy - logqz).shape)
    #ELBO = torch.mean(torch.logsumexp(logpxobsgivenz + logpz + logpy - logqy - logqz,0))
    neg_bound = - ELBO

    #print(neg_bound)
    approx_bound = ELBO
    #print(-torch.mean(logpxobsgivenz).item(), torch.mean(k_l).item())
    #print(logpxobsgivenz, k_l)
    #print(neg_bound)
    return neg_bound, approx_bound, torch.mean(logpxobsgivenz), torch.mean(logqz), torch.mean(logqy)

def mixture_loss(iota_x, mask, p_z, means, scales, logits, decoder, device, d, K=1, K_z=1, data='mnist', beta=1, evaluate=False, full = None, iaf=False, autoregressive_nn=None, autoregressive_nn2= None):
    ##Not needed I guess
    mix = td.categorical.Categorical(torch.ones(d,).cuda())  
    num_components = logits.shape[1]
    batch_size = iota_x.shape[0]
    #print(iota_x)
    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1]) 
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])  

    #print(iota_x_)
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]
    mp = len(iota_x[~mask])
    
    if data=='svhn':
        sigma_decoder = decoder.get_parameter("log_sigma")
    #iota_x_[~mask_] =  p_xm.rsample([K]).reshape(-1)   
    q_z = ReparameterizedNormalMixture1d(logits, means, torch.nn.Softplus()(scales))

    k_l = torch.zeros(K,batch_size)
    zgivenx = torch.zeros(K*batch_size,d)

    #print(iota_x_)
    if iaf:
        #zgivenx, k_l, flow_dist = inverse_autoregressive_flow(q_zgivenxobs, p_z, autoregressive_nn, d, K)
        transform = AffineAutoregressive(autoregressive_nn).cuda()
        transform2 = AffineAutoregressive(autoregressive_nn2).cuda()

        pyro.module("my_transform", transform)  
        flow_dist = pyro.distributions.torch.TransformedDistribution(q_z, [transform, transform2]) #, transform2, transform3, transform4, transform5
        zgivenx = flow_dist.rsample([K])  
        logqz = flow_dist.log_prob(zgivenx).reshape(K,batch_size)
        logpz = p_z.log_prob(zgivenx).reshape(K,batch_size)    
        k_l = torch.mean(logqz - logpz)
    else:
        zgivenx = q_z.rsample([K*K_z])
        logqz = q_z.log_prob(zgivenx).reshape(K,batch_size)
        logpz = p_z.log_prob(zgivenx).reshape(K,batch_size)
        k_l = torch.mean(logqz - logpz)

    zgivenx_flat = zgivenx.reshape([K*K_z*batch_size,d])
    all_logits_obs_model = decoder.forward(zgivenx_flat)
    data_flat = iota_x_.reshape([-1,1])
    tiledmask = mask_.reshape([K*batch_size,channels*p*q]).cuda()

    if evaluate:
        full_ = torch.Tensor.repeat(full,[K,1,1,1]) 
        data_flat = full_.reshape([-1,1])

    if data=='mnist':
        all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    else:
        all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale = sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
        #print(all_log_pxgivenz_flat)

    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size]) 
    logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 
    neg_bound = - (torch.mean(logpxobsgivenz - beta*logqz + beta*logpz) )  # + beta*td.Categorical(logits=logits).entropy() - beta*td.Categorical(logits=logits).entropy().detach() 
    approx_bound = torch.mean(logpz + logpxobsgivenz)

    if evaluate:
        neg_bound =  - torch.mean(torch.logsumexp(logpxmissgivenz + logpz - logqz,0))

    if iaf:
        return neg_bound, approx_bound, -torch.mean(logpxobsgivenz), torch.mean(k_l), flow_dist
    else:
        return neg_bound, approx_bound, -torch.mean(logpxobsgivenz), torch.mean(k_l)



def xm_loss_(iota_x, sampled, mask, p_z, xm_params, encoder, decoder , device, d, beta, K=1, K_z=1, train=True, epoch=0):
    batch_size = iota_x.shape[0]
    p = iota_x.shape[2]
    q = iota_x.shape[3]
    #p=28
    #q=28

    mp = len(iota_x[~mask])
    p_xm = td.continuous_bernoulli.ContinuousBernoulli(logits=xm_params.reshape([-1,1]))

    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1])       #.reshape([-1,1])       
    #iota_x_ = torch.Tensor.repeat(sampled,[K,1,1,1])  
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])         #.reshape([-1,1]) 

    if train==True: #and epoch>=1
        iota_x_[~mask_] =  p_xm.rsample([K]).reshape(-1)            ## plot image and check here...

    out_encoder = encoder.forward(iota_x_) 
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
    zgivenx = q_zgivenxobs.rsample([K_z])
    zgivenx_flat = zgivenx.reshape([K*K_z*batch_size,d])
    #print(zgivenx_flat.shape)                           # sample one z
    all_logits_obs_model = decoder.forward(zgivenx_flat)
    #print(all_logits_obs_model)
    #data_flat = torch.Tensor.repeat(data,[K,1]).reshape([-1,1])
    data_flat = iota_x_.reshape([-1,1])
    #tiledmask = torch.Tensor.repeat(mask_,[K,1])        #reshape([batch_size,28*28])
    tiledmask = mask_.reshape([K*batch_size,28*28]).cuda()

    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    #print(all_log_pxgivenz_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p*q])

    #logpxobsgivenz = torch.sum(all_log_pxgivenz,1).reshape([K,batch_size])          ####*tiledmask??????   *tiledmask
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size]) 
    logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 

    logpxobsgivenz_ = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])            ##

    logpz = p_z.log_prob(zgivenx).reshape([K,batch_size])
    ##print its shape to see what shape data[~mask] should be in
    logq = q_zgivenxobs.log_prob(zgivenx).reshape([K,batch_size])
    #print(iota_x_[~mask_].reshape(K*batch_size,mp))

    k_l1 = torch.distributions.kl.kl_divergence(q_zgivenxobs, p_z)
    #print(k_l1)
    k_l = latent_loss(out_encoder[...,:d], torch.nn.Softplus()(out_encoder[...,d:]))
    #print(k_l.shape)
    #print(xm_params.size(), iota_x_[~mask_].shape)
    miss_log_px_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=torch.Tensor.repeat(xm_params,[K,1,1,1]).reshape([-1,1])).log_prob(iota_x_[~mask_].reshape(-1,1))
    miss_log_px = miss_log_px_flat.reshape([K*batch_size,mp])
    logqm = torch.sum(miss_log_px,1).reshape([K,batch_size])    

    #q_entropy = torch.sum(td.continuous_bernoulli.ContinuousBernoulli(logits=xm_params.reshape([-1,1])).entropy())
    q_entropy = torch.tensor(0).to(device,dtype = torch.float)
    for x in xm_params: 
        q_entropy += entropy(x,device) 

    #logqm = p_xm.log_prob(iota_x_[~mask_].reshape(K*batch_size,mp)) 
    #print(logpxobsgivenz.shape, logpz.shape, logq.shape, logqm.shape)
    #print(q_entropy)
    #print(logpxobsgivenz.item(),logpz.item(),logq.item(), logqm.item(), q_entropy.item())
    #print(torch.logsumexp(logpxobsgivenz + logpz - logq - logqm,0))
    #neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq + q_entropy,0))
    #print(k_l.shape, logpxobsgivenz.shape, q_entropy.shape)
    #alpha-annelaing
    neg_bound = - (torch.mean(logpxobsgivenz + logpxmissgivenz - k_l) + beta*q_entropy) 
    #print(-torch.mean(logpxobsgivenz + logpxmissgivenz).item(), torch.mean(k_l).item(), -torch.mean(q_entropy).item())
    #beta-annealing
    #neg_bound = - (torch.mean(logpxobsgivenz + (beta)*logpxmissgivenz - k_l) + (beta)* q_entropy)

    #neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq - logqm,0))
    #print(all_logits_obs_model.shape)
    #print("All components of loss ---")
    #print(torch.sum(logpxobsgivenz), torch.sum(logpz), torch.sum(logq), torch.prod(torch.sigmoid(logqm)))
    #print("loss :", neg_bound.item())
    #print(logqm,torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0)))
    #log_like = torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0))
    #approx_bound = -torch.mean(torch.logsumexp(logpxobsgivenz_ + logpz - logq,0))
    approx_bound = torch.mean(logpz + logpxobsgivenz + logpxmissgivenz)
    #print(torch.mean(logpz), torch.mean(logpxobsgivenz_))
    return neg_bound, approx_bound, -torch.mean(logpxobsgivenz + logpxmissgivenz), torch.mean(k_l), -torch.mean(q_entropy)


def xm_loss_q(iota_x,  sampled, mask, p_z, z_params, xm_params, encoder, decoder , device, d, beta, K=1, K_z=1, train=True, epoch=0, data='mnist', scales=None, evaluate=False, full=None):
    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]

    mp = len(iota_x[~mask])

    if data=='svhn':
        #sigma_decoder = decoder.get_parameter("log_sigma")
        p_xm = td.Normal(loc = xm_params.to(device,dtype = torch.float).reshape([-1,1]), scale =  scales.exp().to(device,dtype = torch.float).reshape([-1,1]))
        sigma_decoder = decoder.get_parameter("log_sigma")
    else:
        p_xm = td.continuous_bernoulli.ContinuousBernoulli(logits=xm_params.reshape([-1,1]))

    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1])       #.reshape([-1,1])       
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])         #.reshape([-1,1]) 

    if train==True: #and epoch>=1
        iota_x_[~mask_] =  p_xm.rsample([K]).reshape(-1)            ## plot image and check here...

    if evaluate ==True:
        logqxm = torch.zeros(K,mp).to(device,dtype = torch.float)
        for i in range(K):
            all_log_pz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=xm_params.reshape([-1,1])).log_prob(iota_x_[~mask_][i*mp:(i+1)*mp].reshape([-1,1]))
            all_log_pz = all_log_pz_flat.reshape([1,mp])
            logqxm[i] = all_log_pz

        logqx_m = torch.sum(logqxm,1).reshape([K,batch_size]) 
        full_ = torch.Tensor.repeat(full,[K,1,1,1]) 
        

    out_encoder = encoder.forward(iota_x_) 
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)
    z_params = out_encoder
    #q_zgivenxobs = td.Independent(td.Normal(loc=z_params[...,:d],scale=torch.nn.Softplus()(z_params[...,d:])),1)
    zgivenx = q_zgivenxobs.rsample([K_z]) #K with z_params and without K if out_encoder is used

    logqz = q_zgivenxobs.log_prob(zgivenx).reshape(K,batch_size)
    logpz = p_z.log_prob(zgivenx).reshape(K,batch_size)


    zgivenx_flat = zgivenx.reshape([K*K_z*batch_size,d])
    all_logits_obs_model = decoder.forward(zgivenx_flat)

    data_flat = iota_x_.reshape([-1,1])
    if evaluate:
        data_flat = full_.reshape([-1,1])
    tiledmask = mask_.reshape([K*batch_size,channels*p*q]).cuda()

    if data=='mnist':
        #print(all_logits_obs_model.reshape([-1,1]).shape, data_flat.shape)
        all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
        #print(all_log_pxgivenz_flat.shape)
    else:
        all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale =  sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)

    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,channels*p*q])
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size]) 
    logpxmissgivenz = torch.sum(all_log_pxgivenz*(~tiledmask),1).reshape([K,batch_size]) 

    logpz = p_z.log_prob(zgivenx).reshape([K,batch_size])
    logq = q_zgivenxobs.log_prob(zgivenx).reshape([K,batch_size])

    k_l = latent_loss(z_params[...,:d], torch.nn.Softplus()(z_params[...,d:]))

    #miss_log_px_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=torch.Tensor.repeat(xm_params,[K,1,1,1]).reshape([-1,1])).log_prob(iota_x_[~mask_].reshape(-1,1))
    #miss_log_px = miss_log_px_flat.reshape([K*batch_size,mp])
    #logqm = torch.sum(miss_log_px,1).reshape([K,batch_size])    

    q_entropy = torch.tensor(0).to(device,dtype = torch.float)

    ### Change entropy for normal distribution of svhn .....
    if data=='mnist':
        for x in xm_params: 
            q_entropy += entropy(x,device) 
    else:
        q_entropy = torch.sum(p_xm.entropy())
        #print(q_entropy)

    #q_entropy = q_entropy/mp
    if evaluate==False:
        neg_bound = - (torch.mean(logpxobsgivenz + logpxmissgivenz - k_l) + beta*q_entropy) 
    else:
        neg_bound = - torch.logsumexp(logpxmissgivenz + logpz - logqx_m - logqz, 0) 

    #print(-torch.mean(logpxobsgivenz + logpxmissgivenz).item(), torch.mean(k_l).item(), -torch.mean(q_entropy).item())
    approx_bound = torch.mean(logpz + logpxobsgivenz + logpxmissgivenz)
    return neg_bound, approx_bound, -torch.mean(logpxobsgivenz + logpxmissgivenz), torch.mean(k_l), -torch.mean(q_entropy)

def mvae_impute(iota_x,mask,encoder,decoder,p_z, d, L=1, with_labels=False, labels= None):
    batch_size = iota_x.shape[0]
    channels = iota_x.shape[1]
    p = iota_x.shape[2]
    q = iota_x.shape[3]
    #iota_x = torch.reshape(iota_x, (batch_size, p*q))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_encoder = encoder.forward(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([L])
    if with_labels:
        #print(zgivenx.shape, labels.shape)
        ##No information in labels
        labels = torch.zeros(1,labels.shape[0],10).to(device,dtype = torch.float)
        #labels = labels.reshape(1,labels.shape[0],10)
        #labels = labels.repeat(L)
        zgivenx_y = torch.cat((zgivenx,labels),2)
        #print(zgivenx_y.shape)
        zgivenx_flat = zgivenx_y.reshape([L*batch_size,d+10])
    else:
        zgivenx_flat = zgivenx.reshape([L*batch_size,d])

    all_logits_obs_model = decoder.forward(zgivenx_flat)

    data_flat = iota_x[:,0,:,:].reshape([-1,1]).cuda()
    tiledmask = mask.reshape([batch_size,28*28]).cuda()

    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    #all_log_pxgivenz_flat = td.bernoulli.Bernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,28*28])
    
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model),1)
    #xgivenz = td.Independent(td.bernoulli.Bernoulli(logits=all_logits_obs_model),1)

    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    #print(torch.sum(logpxobsgivenz), torch.sum(logpz), torch.sum(logq))

    #xms = xgivenz.sample().reshape([L,batch_size,28*28])
    #xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
    #xm = torch.sigmoid(all_logits_obs_model).reshape([1,batch_size,28*28])
    xm_logits = all_logits_obs_model.reshape([1,batch_size,28,28])

    return xm_logits, out_encoder
          

def mvae_impute_svhn(iota_x, mask, encoder, decoder, p_z, d, L=1):
    batch_size = iota_x.shape[0]
    p = iota_x.shape[2]
    q = iota_x.shape[3]
    channels = iota_x.shape[1]
    
    #iota_x = torch.reshape(iota_x, (batch_size, p*q))
    sigma_decoder = decoder.get_parameter("log_sigma")

    out_encoder = encoder.forward(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])

    all_logits_obs_model = decoder.forward(zgivenx_flat)

    iota_x_flat = iota_x.reshape(batch_size,channels*p*q)

    data_flat = iota_x.reshape([-1,1]).cuda()
    tiledmask = mask.reshape([batch_size,channels*p*q]).cuda()

    #all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz_flat = td.Normal(loc = all_logits_obs_model.reshape([-1,1]), scale =  sigma_decoder.exp()*(torch.ones(*all_logits_obs_model.shape).cuda()).reshape([-1,1])).log_prob(data_flat)
    #all_log_pxgivenz_flat = td.bernoulli.Bernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,channels*p*q])
    
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model),1)
    #xgivenz = td.Independent(td.bernoulli.Bernoulli(logits=all_logits_obs_model),1)

    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
    #print(torch.sum(logpxobsgivenz), torch.sum(logpz), torch.sum(logq))

    #xms = xgivenz.sample().reshape([L,batch_size,28*28])
    #xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
    #xm = torch.sigmoid(all_logits_obs_model).reshape([1,batch_size,28*28])
    xm_logits = all_logits_obs_model.reshape([batch_size,channels,p,q])
    #imputed_image = torch.sigmoid(xm_logits)
    return xm_logits, out_encoder, sigma_decoder


def xm_impute(iota_x,mask,p_z, p_xm, encoder, decoder, d, L=1):

    batch_size = iota_x.shape[0]
    p=28
    q=28
    #iota_x = torch.reshape(iota_x, (batch_size, p*q))
    #mask = torch.reshape(mask, (batch_size, p*q))

    #iota_x[~mask] = p_xm.rsample([L])

    out_encoder = encoder.forward(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])

    all_logits_obs_model = decoder.forward(zgivenx_flat)

    iota_x_flat = iota_x.reshape(batch_size,28*28)

    data_flat = iota_x.reshape([-1,1]).cuda()
    tiledmask = mask.reshape([batch_size,28*28]).cuda()

    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,28*28])
    
    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    logqm = p_xm.log_prob(iota_x[~mask])

    xgivenz = td.Independent(td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model),1)

    #imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq - logqm,0) # these are w_1,....,w_L for all observations in the batch
    #print(imp_weights, imp_weights.size)
    #xm = xgivenz.sample().reshape([L,batch_size,28*28])
    xm = torch.sigmoid(all_logits_obs_model).reshape([1,batch_size,28*28])
    #xm=torch.einsum('ki,kij->ij', imp_weights, xms) 

    return xm

def generate_samples(p_z, decoder, d, L=1, data="mnist"):
    z = p_z.sample([L])
    z_flat = z.reshape([L,d])
    all_logits_obs_model = decoder.forward(z_flat)

    if data =='svhn':
        xm = all_logits_obs_model.reshape([1,3,32*32])
    else:
        xm = torch.sigmoid(all_logits_obs_model).reshape([1,1,28*28])
    return xm

def generate_samples_with_labels(means, scales, logits_y, decoder, d, L=1, data="mnist", logits=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    q_y = td.Categorical(logits=logits_y)
    y = q_y.sample([L])
    #print(y)
    #print(torch.nn.functional.softmax(logits_y, dim=1))
    if logits is None:
        q_zgiveny = td.Independent(td.Normal(loc=means[y,:], scale=torch.nn.Softplus()(scales[y,:])),1) 
    else:
        q_zgiveny = ReparameterizedNormalMixture1d(logits[y].reshape(1,10), means[y].reshape(1, 10, d), scales[y].exp().reshape(1, 10, d))

    z = q_zgiveny.sample([L])

    y_one_hot = torch.zeros((1, 10))
    y_one_hot[0, y] = 1

    z_flat = z.reshape([L,d])
    y_flat = y_one_hot.reshape([L,10])
    zy_flat = torch.cat((z_flat,y_flat.to(device,dtype = torch.float)),1)

    all_logits_obs_model = decoder.forward(zy_flat)

    if data =='svhn':
        xm = all_logits_obs_model.reshape([1,3,32*32])
    else:
        xm = torch.sigmoid(all_logits_obs_model).reshape([1,1,28*28])
    return xm

def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    num_missing = 1*28*28 - int(mask.sum())

    return np.mean(np.power(xhat-xtrue,2)[~mask])/num_missing

def get_entropy_mixture(logits):
    print("probabilities ---",torch.softmax(logits,dim=1))
    dist = td.categorical.Categorical(logits=logits)
    return dist.entropy()


