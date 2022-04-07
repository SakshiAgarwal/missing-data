from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from plot import *

class BinaryMNIST_Test(Dataset):
	def __init__(self,binarize=False,perc_miss=0):
		test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

		self.images = test_dataset.data 				
		#self.labels = test_data.targets
		self.perc_miss = perc_miss

		##Normalize data to 
		if binarize:
		    self.images[self.images < 127] = 0
		    self.images[self.images >= 127] = 1  #255
		else:
		    self.images = self.images.float()/255

		np.random.seed(1234)

		self.n=self.images.size()[0]
		self.p=self.images.size()[1]
		self.q=self.images.size()[2]

		xmiss = np.copy(self.images).astype(np.float)
		xmiss_flat = xmiss.flatten()
		miss_pattern = [(i)*self.p*self.q+ np.random.choice(self.p*self.q, np.floor(self.p*self.q*perc_miss).astype(np.int), replace=False) for i in range(self.n)]

		miss_pattern = np.asarray(miss_pattern).astype(np.int)
		xmiss_flat[miss_pattern] = np.nan
		xmiss = xmiss_flat.reshape([self.n,self.p,self.q])
		mask = np.isfinite(xmiss).astype(np.float)  # False indicates missing, True indicates observed
		xhat_0 = np.copy(xmiss)
		xhat_0[np.isnan(xmiss)] = 0
		self.images = self.images.reshape(self.n,1,self.p,self.q)
		self.xhats_0 = xhat_0.reshape(self.n,1,self.p,self.q).astype(np.float)
		self.masks = mask.reshape(self.n,1,self.p,self.q)

	def __getitem__(self, idx):
		return self.xhats_0[idx], self.masks[idx], self.images[idx]

	def __len__(self):  
	    return len(self.images)

class BinaryMNIST(Dataset):
	def __init__(self,dataset, perc_miss=0):
		self.images  = dataset.data
		self.n=self.images.size()[0]
		self.p=self.images.size()[1]
		self.q=self.images.size()[2]

		self.images = self.images.reshape(self.n,1,self.p,self.q)
		self.masks = np.ones(np.shape(self.images)).astype(np.bool)

	def __getitem__(self, idx):
		return self.images[idx], self.masks[idx]

	def __len__(self):	
		return len(self.images)

def train_valid_loader(data_dir, batch_size=64, valid_size=0.2, binary_data = False):
	#normalize = transforms.Normalize((0.5), (0.5))

	train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
	valid_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())


	if binary_data:
		train_dataset.data[train_dataset.data<127]=0.0
		train_dataset.data[train_dataset.data>=127]=1.0
		
		valid_dataset.data[train_dataset.data<127]=0.0
		valid_train_dataset.data[train_dataset.data>=127]=1.0
	else:
		train_dataset.data = (train_dataset.data.double()/255)
		valid_dataset.data = (valid_dataset.data.double()/255)

	#num_val = int(np.floor(valid_size * len(dataset)))
	#num_train = len(dataset) - num_val
	
	num_train = len(train_dataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.seed(1234)
	np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST(train_dataset),batch_size=batch_size, sampler=train_sampler)
	valid_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST(valid_dataset),batch_size=1, sampler=valid_sampler)

	print(len(train_loader.dataset), len(valid_loader.dataset))
	
	#train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])
	#train_loader = torch.utils.data.DataLoader(train_set.dataset, 
	#           batch_size=batch_size, shuffle=True)
	#valid_loader = torch.utils.data.DataLoader(val_set.dataset, 
	#            batch_size=batch_size, shuffle=True)

	return train_loader, valid_loader


