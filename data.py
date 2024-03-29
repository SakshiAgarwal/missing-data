from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from plot import *
from sklearn.datasets import load_iris, load_breast_cancer, load_boston 
import pandas as pd

def load_uci_datasets(dataset='iris', missing_for_train = False):
	if dataset=='iris':
		data = load_iris()['data']
		print(data)
	elif dataset=='breast_cancer':
		data = load_breast_cancer(True)[0]
	elif dataset == 'boston':
		data = load_boston(True)[0]
	elif dataset == "red_wine":
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
		data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
	elif dataset == "white_wine":
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
		data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
	elif dataset =='banknote':
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
		data = np.array(pd.read_csv(url, low_memory=False, sep=','))[:,0:4]

	np.random.seed(1234)
	#print(np.mean(data,0), np.std(data,0) )
	xfull = (data - np.mean(data,0))/np.std(data,0)
	#print(np.shape(xfull))
	#print(xfull[-1])
	np.random.shuffle(xfull)
	#print(xfull[-1])

	print(np.shape(xfull))

	n = np.shape(xfull)[0] # number of observations
	p = np.shape(xfull)[1] # number of features

	split1 = int(0.65*n)
	split2 = int(0.8*n)
	#print(split1, split2)
	trainset = xfull[:split1]
	validset = xfull[split1:split2]
	testset = xfull[split2:]

	n_test = np.shape(testset)[0]
    
	##For missing data in testset
	perc_miss = 0.5 # 50% of missing data
	xhat_0 = np.copy(testset)
	mask = np.ones((n_test, p))
    
	for i in range(n_test):
		miss_pattern = np.random.choice(p, np.floor(p*perc_miss).astype(np.int), replace=False)
		#print(miss_pattern)
		for j in range(len(miss_pattern)):
			xhat_0[i,miss_pattern[j]] = 0
			mask[i,miss_pattern[j]] = 0  
		#print(xhat_0[i], testset[i], mask[i])

	if missing_for_train:
		train_miss = np.copy(trainset)
		n_train = np.shape(trainset)[0]
		mask_train = np.ones((n_train, p))
		for i in range(n_train):
			miss_pattern = np.random.choice(p, np.floor(p*perc_miss).astype(np.int), replace=False)
			#print(miss_pattern)
			for j in range(len(miss_pattern)):
				train_miss[i,miss_pattern[j]] = 0
				mask_train[i,miss_pattern[j]] = 0  
			#print(train_miss[i], trainset[i], mask_train[i])

		valid_miss = np.copy(validset)
		n_valid = np.shape(validset)[0]
		mask_valid = np.ones((n_valid, p))
		for i in range(n_valid):
			miss_pattern = np.random.choice(p, np.floor(p*perc_miss).astype(np.int), replace=False)
			#print(miss_pattern)
			for j in range(len(miss_pattern)):
				valid_miss[i,miss_pattern[j]] = 0
				mask_valid[i,miss_pattern[j]] = 0  
			#print(valid_miss[i], validset[i], mask_valid[i])               

   
	if missing_for_train:
		return train_miss, mask_train, valid_miss, mask_valid, xhat_0, mask
	else:
		return trainset, validset, xhat_0, mask, testset


def train_valid_loader(data_dir, batch_size=64, valid_size=0.2, binary_data = False, return_labels=False, ispatches=False, top_half=False):
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
	#print(num_train)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.seed(1234)
	np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST(train_dataset, patches=ispatches, top_half=top_half, return_labels = return_labels),batch_size=batch_size, sampler=train_sampler)
	valid_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST(valid_dataset, patches =ispatches, top_half=top_half, return_labels = return_labels),batch_size=1, sampler=valid_sampler)
	return train_loader, valid_loader

class SVHN_Test(Dataset):
	def __init__(self, perc_miss=0, top_half=False, patches=False, noise=False, std=0):
		test_dataset = datasets.SVHN(root='data', split='test',  download=True, transform=transforms.ToTensor())
		print(test_dataset.data.shape, test_dataset.data[:1000].shape)        
		self.images = test_dataset.data[:1000] 				
		self.perc_miss = perc_miss
		##Normalize data to 
		print(self.images.shape)
		std = [0.2023, 0.1994, 0.2010]
		mean = [0.4914, 0.4822, 0.4465]
		std = [0.5, 0.5, 0.5]
		mean = [0.5, 0.5, 0.5]
		self.images = (self.images/255)
		self.images[:,0,:,:] = (self.images[:,0,:,:]- mean[0])/std[0]
		self.images[:,1,:,:] = (self.images[:,1,:,:]- mean[1])/std[1]
		self.images[:,2,:,:] = (self.images[:,2,:,:]- mean[2])/std[2]

		np.random.seed(1234)
		#print(np.max(self.images), np.min(self.images))

		self.n=self.images.shape[0]
		self.channels = self.images.shape[1]
		self.p=self.images.shape[2]
		self.q=self.images.shape[3]

		xmiss = np.copy(self.images).astype(np.float)
		xmiss_flat = xmiss.flatten()

		if patches: 
			num_patches = 2 #1
			patch_size_p = 10 #15
			patch_size_q = 10 #12

			#miss_pattern_x = [np.random.choice((self.p - patch_size), num_patches, replace=False) for i in range(self.n)]
			#miss_pattern_y = [(i)*self.p*self.q + self.p*np.random.choice((self.p-patch_size), num_patches, replace=False) for i in range(self.n)]

			miss_pattern_x = [np.random.choice((self.p - patch_size_p), num_patches, replace=False) for i in range(self.n)]
			miss_pattern_y = [np.random.choice((self.q - patch_size_q), num_patches, replace=False) for i in range(self.n)]

			#print(miss_pattern_x)
			#miss_pattern_x = np.asarray(miss_pattern_x).astype(np.int)
			#miss_pattern_y = np.asarray(miss_pattern_y).astype(np.int)
			for i in range(self.n):
				for a,b in zip(miss_pattern_y[i], miss_pattern_x[i]):
					start_x = int(a)
					end_x = int(a + patch_size_q)
					start_y = int(b)
					end_y = int(b + patch_size_p)
					xmiss[i,:,start_x: end_x, start_y:end_y] = np.nan

			#xmiss = xmiss_flat.reshape([self.n,self.channels,self.p,self.q])
			mask = np.isfinite(xmiss).astype(np.bool)  # False indicates missing, True indicates observed
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0

		if top_half:
			mask = np.zeros((1,3,32,32))
			mask[:,: ,16:,:] = 1
			mask[:,: ,:,:6] = 1
			mask[:,: ,:,26:] = 1
			#mask[:,: ,16:,:] = 1
			##Right half missing : mask[:,:14] = 1
			##Bottom half missing : mask[:14,:] = 1
			mask = mask.astype(np.bool)
			mask = np.tile(mask, (self.n,1,1,1))
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0


		self.xhats_0 = xhat_0.reshape(self.n,self.channels,self.p,self.q).astype(np.float)
		self.masks = mask.reshape(self.n,self.channels,self.p,self.q).astype(np.bool)
		self.images = self.images.reshape(self.n,self.channels,self.p,self.q)


	def __getitem__(self, idx):
		return self.xhats_0[idx], self.masks[idx], self.images[idx]

	def __len__(self):  
	    return len(self.images)



class BinaryMNIST_Test(Dataset):
	def __init__(self,binarize=False,perc_miss=0,top_half=False,patches=False, noise=False,std=0, return_labels=False):
		test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
		self.images = test_dataset.data 				
		self.labels = test_dataset.targets        
		print(self.labels.shape[0])
		max_label = 9
		b_ = np.zeros((self.labels.shape[0], 10))
		b_[np.arange(self.labels.shape[0]), self.labels] = 1

		self.labels = b_
		print(self.labels.shape)
		self.perc_miss = perc_miss

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

		if perc_miss:
			miss_pattern = [(i)*self.p*self.q + np.random.choice(self.p*self.q, np.floor(self.p*self.q*perc_miss).astype(np.int), replace=False) for i in range(self.n)]
			self.miss_pattern = np.asarray(miss_pattern).astype(np.int)
			#print(miss_pattern)
			xmiss_flat[miss_pattern] = np.nan
			xmiss = xmiss_flat.reshape([self.n,self.p,self.q])
			mask = np.isfinite(xmiss).astype(np.bool)  # False indicates missing, True indicates observed
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0

		if noise:
			#std = 10
			miss_pattern = [(i)*self.p*self.q + np.random.choice(self.p*self.q, np.floor(self.p*self.q*perc_miss).astype(np.int), replace=False) for i in range(self.n)]
			self.miss_pattern = np.asarray(miss_pattern).astype(np.int)
			#print(miss_pattern)
			xmiss_flat[miss_pattern] = np.nan
			xmiss = xmiss_flat.reshape([self.n,self.p,self.q])
			mask = np.isfinite(xmiss).astype(np.bool)  # False indicates missing, True indicates observed
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] += std*np.random.randn(*xhat_0[~mask].shape) 
			xhat_0[mask] = np.random.uniform(0,1, *xhat_0[mask].shape)

			print("len of gaussian noise: ", len(xhat_0[~mask]))
			#quit()
			#xhat_0 = np.copy(self.images).astype(np.float) 
			#print("image --- ")
			#print(xhat_0[0])
			#xhat_0 += std*np.random.randn(*xhat_0.shape) 
			#print("noisy image -- ")
			#xhat_0 = np.clip(xhat_0, 0, 1)
			print(xhat_0[0])

		if top_half:
			mask = np.zeros((28,28))
			mask[14:,:] = 1
			##Right half missing : mask[:,:14] = 1
			##Bottom half missing : mask[:14,:] = 1
			mask = mask.astype(np.bool)
			mask = np.tile(mask, (self.n,1,1))
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0

		if patches: 
			num_patches = 2
			patch_size = 10

			miss_pattern_x = [np.random.choice((self.p-patch_size), num_patches, replace=False) for i in range(self.n)]
			miss_pattern_y = [(i)*self.p*self.q + self.p*np.random.choice((self.p-patch_size), num_patches, replace=False) for i in range(self.n)]
			#print(miss_pattern_x)
			#miss_pattern_x = np.asarray(miss_pattern_x).astype(np.int)
			#miss_pattern_y = np.asarray(miss_pattern_y).astype(np.int)

			for p in range(patch_size):
				#start = []
				#end = []
				for i in range(self.n):
					for a,b in zip(miss_pattern_y[i], miss_pattern_x[i]):
						#print(a,b)
						start = int(a + (p)*28 + b)
						end = int(a + (p)* 28 + b + patch_size)
						xmiss_flat[start: end] = np.nan
					#exit()
				#start =  [ int((a + p)* 28 + b) for a, b in zip(miss_pattern_y, miss_pattern_x)]
				#end = [ int((a + p)* 28 + b + patch_size) for a, b in zip(miss_pattern_y, miss_pattern_x)]
				#print((miss_pattern_y+p)*28 + miss_pattern_x)
				#print(end, p)
			xmiss = xmiss_flat.reshape([self.n,self.p,self.q])
			mask = np.isfinite(xmiss).astype(np.bool)  # False indicates missing, True indicates observed
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0

		if noise:
			mask = np.zeros((self.n,28,28))

		#print(np.sum(np.isnan(xmiss) == True))
		#print(np.isnan(xmiss))
		#xhat_0[np.isnan(xmiss)] = 0
		#print(np.sum(255*xhat_0[0]), perc_miss)

		self.xhats_0 = xhat_0.reshape(self.n,1,self.p,self.q).astype(np.float)
		self.masks = mask.reshape(self.n,1,self.p,self.q).astype(np.bool)
		self.images = self.images.reshape(self.n,1,self.p,self.q)

		if return_labels:
			#print(np.shape(b_))
			b = b_.reshape(self.images.shape[0], max_label + 1, 1, 1)
			b = np.repeat(b, self.p, axis=2)
			b = np.repeat(b, self.q, axis=3)
			#print(b.shape)
			self.images = np.concatenate((self.images, b), axis=1)
			#print(self.images.shape)

	def __getitem__(self, idx):
		return self.xhats_0[idx], self.masks[idx], self.images[idx], self.labels[idx]

	def __len__(self):  
	    return len(self.images)

class BinaryMNIST(Dataset):
	def __init__(self,dataset, patches=False, top_half = False,return_labels=False):
		self.images  = dataset.data
		self.labels = dataset.targets
		self.n=self.images.size()[0]
		self.p=self.images.size()[1]
		self.q=self.images.size()[2]
		self.images = self.images.reshape(self.n,1,self.p,self.q)
		#print(self.images.shape)
		mask = np.ones(np.shape(self.images)).astype(np.bool)

		channels = 1
		max_label = self.labels.max()
		b = np.zeros((self.labels.shape[0], self.labels.max() + 1))
		b[np.arange(self.labels.shape[0]), self.labels] = 1
		self.labels = b

		xmiss = np.copy(self.images).astype(np.float)
		xmiss_flat = xmiss.flatten()
		if patches: 
			num_patches = 2
			patch_size = 10
			miss_pattern_x = [np.random.choice((self.p-patch_size), num_patches, replace=False) for i in range(self.n)]
			miss_pattern_y = [(i)*self.p*self.q + self.p*np.random.choice((self.p-patch_size), num_patches, replace=False) for i in range(self.n)]
			for p in range(patch_size):
				#start = []
				#end = []
				for i in range(self.n):
					for a,b in zip(miss_pattern_y[i], miss_pattern_x[i]):
						#print(a,b)
						start = int(a + (p)*28 + b)
						end = int(a + (p)* 28 + b + patch_size)
						xmiss_flat[start: end] = np.nan
			xmiss = xmiss_flat.reshape([self.n,self.p,self.q])
			mask = np.isfinite(xmiss).astype(np.bool)  # False indicates missing, True indicates observed
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0
		elif top_half:
			mask = np.zeros((28,28))
			mask[14:,:] = 1
			##Right half missing : mask[:,:14] = 1
			##Bottom half missing : mask[:14,:] = 1
			mask = mask.astype(np.bool)
			mask = np.tile(mask, (self.n,1,1)).reshape([self.n,1,28,28])
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0
		else:
			xhat_0 = np.copy(self.images).astype(np.float)

		if return_labels:
			b = b.reshape(self.images.shape[0], max_label + 1, 1, 1)
			b = np.repeat(b, self.p, axis=2)
			b = np.repeat(b, self.q, axis=3)
			#print(b.shape)
			self.images = np.concatenate((self.images, b), axis=1)
			channels = 11
			#print(self.images.shape)

		self.xhats_0 = xhat_0.reshape(self.n,1,self.p,self.q).astype(np.float)
		self.masks = mask.reshape(self.n,1,self.p,self.q).astype(np.bool)
		self.images = self.images.reshape(self.n,channels,self.p,self.q)

	def __getitem__(self, idx):
		return self.xhats_0[idx], self.masks[idx], self.images[idx],  self.labels[idx]

	def __len__(self):	
		return len(self.images)

def train_valid_loader(data_dir, batch_size=64, valid_size=0.2, binary_data = False, return_labels=False, ispatches=False, top_half=False):
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
	#print(num_train)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.seed(1234)
	np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST(train_dataset, patches=ispatches, top_half=top_half, return_labels = return_labels),batch_size=batch_size, sampler=train_sampler)
	valid_loader = torch.utils.data.DataLoader(dataset=BinaryMNIST(valid_dataset, patches =ispatches, top_half=top_half, return_labels = return_labels),batch_size=batch_size, sampler=valid_sampler)
	return train_loader, valid_loader

def get_sample_digit(data_dir, digit, file):
	train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())

	images = train_dataset.data 				
	labels = train_dataset.targets

	images_filter = images[np.where((labels == digit))].double()/255

	plot_image(images_filter[0], file= file + str(digit)  + '.png')
	return images_filter[0:50].reshape(50,1,28,28)



class SVHM(Dataset):
	def __init__(self,dataset, perc_miss=0, patches=False, top_half=False):
		self.images  = dataset.data
		#print(self.images)

		#print(self.images.shape)
		self.n=self.images.shape[0]
		self.p=self.images.shape[2]
		self.q=self.images.shape[3]
		self.channels = self.images.shape[1]

		self.images = self.images.reshape(self.n,3,self.p,self.q)
		self.masks = np.ones(np.shape(self.images)).astype(np.bool)

		xmiss = np.copy(self.images).astype(np.float)
		xmiss_flat = xmiss.flatten()

		if patches: 
			num_patches = 1 #2
			patch_size_p = 15
			patch_size_q = 12

			miss_pattern_x = [np.random.choice((self.p - patch_size_p), num_patches, replace=False) for i in range(self.n)]
			miss_pattern_y = [np.random.choice((self.q - patch_size_q), num_patches, replace=False) for i in range(self.n)]

			for i in range(self.n):
				for a,b in zip(miss_pattern_y[i], miss_pattern_x[i]):
					start_x = int(a)
					end_x = int(a + patch_size_q)
					start_y = int(b)
					end_y = int(b + patch_size_p)
					xmiss[i,:,start_x: end_x, start_y:end_y] = np.nan

			mask = np.isfinite(xmiss).astype(np.bool)  # False indicates missing, True indicates observed
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0
			self.masks = mask.reshape(self.n,self.channels,self.p,self.q).astype(np.bool)
		elif top_half:
			mask = np.zeros((1,3,32,32))
			#mask[:,: ,int(32/2):,:] = 1
			mask[:,: ,16:,:] = 1
			mask[:,: ,:,:6] = 1
			mask[:,: ,:,26:] = 1
			##Right half missing : mask[:,:14] = 1
			##Bottom half missing : mask[:14,:] = 1
			mask = mask.astype(np.bool)
			mask = np.tile(mask, (self.n,1,1,1))
			xhat_0 = np.copy(self.images).astype(np.float)
			xhat_0[~mask] = 0
			self.masks = mask.reshape(self.n,self.channels,self.p,self.q).astype(np.bool)
		else:
			xhat_0 = np.copy(self.images).astype(np.float)
        
		self.xhats_0 = xhat_0.reshape(self.n,self.channels,self.p,self.q).astype(np.float)
		
		self.images = self.images.reshape(self.n,self.channels,self.p,self.q)

	def __getitem__(self, idx):
		return self.xhats_0[idx] , self.masks[idx], self.images[idx]

	def __len__(self):	
		return len(self.images)


def train_valid_loader_svhn(data_dir, batch_size=64, valid_size=0.2, binary_data = False, top_half=False, patches=False):
	#normalize = transforms.Normalize((0.5), (0.5))

	train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transforms.ToTensor())
	valid_dataset = datasets.SVHN(root=data_dir, split='train',  download=True, transform=transforms.ToTensor())
	transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
	svhn = datasets.SVHN(root=data_dir, split='train',  download=True, transform=transform)
	#print(np.max(svhn.data), np.min(svhn.data))
	#exit()    

	train_dataset.data = (train_dataset.data/255)#*2 - 1
	valid_dataset.data = (valid_dataset.data/255)#*2 - 1
	std = [0.2023, 0.1994, 0.2010]
	mean = [0.4914, 0.4822, 0.4465]
	std = [0.5, 0.5, 0.5]
	mean = [0.5, 0.5, 0.5]
	train_dataset.data[:,0,:,:] = (train_dataset.data[:,0,:,:] - mean[0])/std[0]
	train_dataset.data[:,1,:,:] = (train_dataset.data[:,1,:,:] - mean[1])/std[1]
	train_dataset.data[:,2,:,:] = (train_dataset.data[:,2,:,:] - mean[2])/std[2]
	print(np.max(train_dataset.data), np.min(train_dataset.data))
	valid_dataset.data[:,0,:,:] = (valid_dataset.data[:,0,:,:] - mean[0])/std[0]
	valid_dataset.data[:,1,:,:] = (valid_dataset.data[:,1,:,:] - mean[1])/std[1]
	valid_dataset.data[:,2,:,:] = (valid_dataset.data[:,2,:,:] - mean[2])/std[2]

	#num_val = int(np.floor(valid_size * len(dataset)))
	#num_train = len(dataset) - num_val
	
	#num_train = len(train_dataset)
	#indices = list(range(num_train))
	#split = int(np.floor(valid_size * num_train))
	#np.random.seed(1234)
	#np.random.shuffle(indices)

	#train_idx, valid_idx = indices[split:], indices[:split]

	#train_sampler = SubsetRandomSampler(train_idx)
	#valid_sampler = SubsetRandomSampler(valid_idx)

	num_train = len(train_dataset)
	#print(num_train)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.seed(1234)
	np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(dataset=SVHM(train_dataset,patches=patches, top_half=top_half), batch_size=batch_size, sampler=train_sampler) 
	valid_loader = torch.utils.data.DataLoader(dataset=SVHM(valid_dataset,patches=patches, top_half=top_half), batch_size=batch_size, sampler=valid_sampler)

	print("length of train set", len(train_dataset))

	#train_loader = torch.utils.data.DataLoader(dataset=SVHM(train_dataset),batch_size=batch_size)
	#valid_loader = torch.utils.data.DataLoader(dataset=SVHM(valid_dataset),batch_size=1)

	#print(len(train_loader.dataset), len(valid_loader.dataset))
	
	#train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])
	#train_loader = torch.utils.data.DataLoader(train_dataset.data, batch_size=batch_size, shuffle=True)
	#valid_loader = torch.utils.data.DataLoader(valid_dataset.data, batch_size=batch_size, shuffle=True)

	return train_loader, valid_loader


