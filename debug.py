import matplotlib.pyplot as plt
import numpy as np
from data import *
import torch
import torch.distributions as td

#torch.set_default_tensor_type(torch.DoubleTensor)

#train_loader, val_loader = train_valid_loader(data_dir ="data",binary_data = False)

#for data in train_loader:
#	b_data, mask = data
#	print("in the loop")
#	print(b_data[0],mask[0])
#	plot_image(np.squeeze(b_data[0]))
#	exit()

xcoord=[]
ycoord=[]
zcoord = []

def mean(x):
	if x != 0.5:
		return x/(2*x-1) + 1/(2*np.arctanh(1-2*x))
	else:
		return 0.5


def log_C(x):
	if x != 0.5:
		return (2/(1-2*x)*np.arctanh(1-2*x))
	else:
		return 2

def entropy(x):
	if x ==0:
		return 0
	else: 
		return 1 -  x/(1-np.exp(-x)) -np.log(x/(np.exp(x)-1))

for x in np.linspace(-100,100,10000):
	a = torch.tensor(x, dtype=torch.float64)
	y = td.continuous_bernoulli.ContinuousBernoulli(logits=a).entropy()
	#print(a.item(), y.item())
	z = td.continuous_bernoulli.ContinuousBernoulli(logits=a).mean
	xcoord.append(a.item())
	ycoord.append(y.item())
	zcoord.append(z.item())

plt.plot(xcoord,ycoord)
plt.show()
plt.savefig("results/entropy.png")
plt.close()

plt.plot(xcoord,zcoord)
plt.show()
plt.savefig("results/mean.png")
plt.close()

xcoord=[]
ycoord=[]
for a in np.linspace(-100,100,10000):
	y = entropy(a)
	print(a, y)
	#z = td.continuous_bernoulli.ContinuousBernoulli(logits=a).mean
	xcoord.append(a)
	ycoord.append(y)

plt.plot(xcoord,ycoord)
plt.show()
plt.savefig("results/entropy.png")
plt.close()
