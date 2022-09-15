##backup code




def get_train_valid_loader(data_dir, batch_size=64, valid_size=0.2, binary_data = False):
	#normalize = transforms.Normalize((0.5), (0.5))

	train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
	valid_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())

	if binary_data:
		print(train_dataset.data[0])
		#train_dataset.data[train_dataset.data<127] = 0.0
		train_dataset.data[train_dataset.data<127]=0.0
		train_dataset.data[train_dataset.data>=127]=1.0
		print(train_dataset.data[0])
		valid_dataset.data[train_dataset.data >= 127] = 1.0
		print(train_dataset.data.dtype)
	else:
		print(train_dataset.data.dtype)
		print(train_dataset.data[1])
		train_dataset.data = (train_dataset.data.double()/255)
		print(train_dataset.data[1])
		plot_image(train_dataset.data[1])
		valid_dataset.data = (valid_dataset.data.double()/255)
		print(train_dataset.data.dtype)
		#torch.flatten(train_dataset, start_dim=1)
		#torch.flatten(valid_dataset, start_dim=1)

	#plot_image(train_dataset.data[0])
	#print(train_dataset.data[0])
	#plot_image(valid_dataset.data[0])
	#print(valid_dataset.data[0])

	num_train = len(train_dataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))

	np.random.seed(1234)
	np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(train_dataset, 
	            batch_size=batch_size, sampler=train_sampler)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, 
	            batch_size=batch_size, sampler=valid_sampler)

	return train_loader, valid_loader


class BinaryMNIST_(Dataset):
    def __init__(self,train=True, binarize=True, perc_miss=0, valid_size=0.2):
        if(train):
            train_dataset = torchvision.datasets.MNIST(root='data',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)
            self.images  = train_dataset.data ##using datasets.MNIST
            self.images = self.images[:int(0.8*self.images.size()[0]), :, :]
        else:
            test_dataset = torchvision.datasets.MNIST(root='data',
                        train=False,
                        transform=transforms.ToTensor())
            self.images = test_dataset.data #self.labels = test_data.targets
            self.images = self.images[:int(0.2*self.images.size()[0]), :, :]
            print("Validation len ", self.images.size())

        if binarize:
            self.images[self.images < 127] = 0
            self.images[self.images >= 127] = 1  #255
        else:
            self.images = self.images/255
            #self.images = 2*self.images - 1

        self.images = self.images.reshape(self.n,1,self.p,self.q)
        self.masks = np.ones(np.shape(self.images)).astype(np.bool)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]
        
    def __len__(self):	
        return len(self.images)


    


