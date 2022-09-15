#import tensorflow_probability.substrates.jax as tfp
#import tensorflow_probability.substrates.jax.distributions as tfd
import torch
import torch.distributions as td
from torch.distributions import Normal
import numpy as np


def inject_grads(final_samples, logits):
    final_samples = final_samples.detach() 

    # Reparameterize logits
    logit_dist = td.Categorical(logits=logits)
    logit_prob = logit_dist.log_prob(final_samples)
    logit_injection = torch.exp(logit_prob - logit_prob.detach()).unsqueeze(-1)
    
    return logit_injection * final_samples

class ReparameterizedNormal1d(td.Categorical):
    '''
    Means/Scales should be of shape: [Batch size, # components, # dimensions]
    Logits should be of shape: [Batch size, # components]
    '''
    def __init__(self, logits):
        super(ReparameterizedNormal1d, self).__init__(td.Categorical(logits=logits))
        self.logits = logits

    def rsample(self, sample_shape=torch.Size(), beta=1):
        sample = self.sample(sample_shape)
        return inject_grads(sample, self.logits)



