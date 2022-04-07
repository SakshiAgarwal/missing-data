import torch
import numpy as np
import torch.distributions as td

def mvae_loss(iota_x, mask, encoder, decoder,p_z, d, K=1):
    batch_size = iota_x.shape[0]
    p = iota_x.shape[2]
    q = iota_x.shape[3]

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
    
    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    #all_log_pxgivenz_flat = td.bernoulli.Bernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p*q])
    #print(all_log_pxgivenz_flat.size(), all_log_pxgivenz.size())

    logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,[1]).reshape([K,batch_size])

    logpz = p_z.log_prob(zgivenx)
    
    logq = q_zgivenxobs.log_prob(zgivenx)

    #print(logpxobsgivenz.size(), logpz.size(), logq.size())
    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
    log_like = torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0))

    return neg_bound, log_like

def xm_loss(iota_x, mask, p_z, p_xm, encoder, decoder, d, K=1, K_z=1):
    batch_size = iota_x.shape[0]
    p = iota_x.shape[2]
    q = iota_x.shape[3]
    mp = len(iota_x[~mask])

    iota_x_ = torch.Tensor.repeat(iota_x,[K,1,1,1])       #.reshape([-1,1])       
    mask_ = torch.Tensor.repeat(mask,[K,1,1,1])         #.reshape([-1,1]) 
    #print(iota_x_.shape, mask_.shape)
    iota_x_[~mask_] =  p_xm.rsample([K]).reshape(-1)
    #print(p_xm.rsample([K]).shape, iota_x_.shape)

    out_encoder = encoder.forward(iota_x_) 
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([K_z])
    #print(zgivenx.shape)
    zgivenx_flat = zgivenx.reshape([K*K_z*batch_size,d])
    #print(zgivenx_flat.shape)                           # sample one z
    all_logits_obs_model = decoder.forward(zgivenx_flat)
    #print(all_logits_obs_model)

    #data_flat = torch.Tensor.repeat(data,[K,1]).reshape([-1,1])
    data_flat = iota_x_.reshape([-1,1])
    #tiledmask = torch.Tensor.repeat(mask_,[K,1])        #reshape([batch_size,28*28])

    all_log_pxgivenz_flat = td.continuous_bernoulli.ContinuousBernoulli(logits=all_logits_obs_model.reshape([-1,1])).log_prob(data_flat)
    #print(all_log_pxgivenz_flat)
    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p*q])

    logpxobsgivenz = torch.sum(all_log_pxgivenz,1).reshape([K,batch_size])

    logpz = p_z.log_prob(zgivenx).reshape([K,batch_size])
    
    ##print its shape to see what shape data[~mask] should be in
    logq = q_zgivenxobs.log_prob(zgivenx).reshape([K,batch_size])
    
    logqm = p_xm.log_prob(iota_x_[~mask_].reshape(K*batch_size,mp)) 

    #print(logpxobsgivenz.shape, logpz.shape, logq.shape, logqm.shape)
    #print(logpxobsgivenz,logpz,logq, logqm)
    #print(torch.logsumexp(logpxobsgivenz + logpz - logq - logqm,0))

    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq - logqm,0))

    #print(all_logits_obs_model.shape)
    #print("All components of loss ---")
    #print(torch.sum(logpxobsgivenz), torch.sum(logpz), torch.sum(logq), torch.prod(torch.sigmoid(logqm)))
    #print("loss :", neg_bound)
    #print(logqm,torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0)))
    log_like = torch.mean(torch.logsumexp(logpxobsgivenz + logpz,0))


    return neg_bound, log_like

def mvae_impute(iota_x,mask,encoder,decoder,p_z, d, L=1):
    batch_size = iota_x.shape[0]

    out_encoder = encoder.forward(iota_x)
    q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[...,:d],scale=torch.nn.Softplus()(out_encoder[...,d:])),1)

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L*batch_size,d])

    all_logits_obs_model = decoder.forward(zgivenx_flat)

    iota_x_flat = iota_x.reshape(batch_size,28*28)

    data_flat = iota_x.reshape([-1,1]).cuda()
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
    xm = torch.sigmoid(all_logits_obs_model).reshape([1,batch_size,28*28])

    return xms
                
def xm_impute(iota_x,mask,p_z, p_xm, encoder, decoder, d, L=1):

    batch_size = iota_x.shape[0]

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
   

    
def mse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat-xtrue,2)[~mask])
