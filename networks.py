import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.nn.utils import spectral_norm
#from .spectral_norm import spectral_norm


class simple_encoder(nn.Module):
    def __init__(self, input_size=28*28,hidden_dim=500,drop=0.9,latent_dim=20):
        nn.Module.__init__(self)

        self.layer1 = torch.nn.Linear(input_size, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, 2*latent_dim)

    def forward(self, x, y=torch.tensor(0.)):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x.float()

class simple_decoder(nn.Module):
    def __init__(self, output_size=28*28,hidden_dim=500,drop=0.9,latent_dim=20):
        nn.Module.__init__(self)

        self.layer1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x, y=torch.tensor(0.)):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x.float()
