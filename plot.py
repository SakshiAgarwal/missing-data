import matplotlib.pyplot as plt
import numpy as np
from data import *
from loss import *
#import cv2
import matplotlib.cm as cm
import torch.distributions as td
from scipy.special import expit

def plot_loss_vs_sample_size(mixture_loss_samples, K_samples_ ,save_directory):
	x =  np.zeros((6,len(K_samples_)))
	for i in range(len(K_samples_)):
		x[:,i] = K_samples_[i]
	x = x.reshape(-1)

	for i in range(2):
		y = mixture_loss_samples[i,:,:].reshape(-1)
		plt.scatter(x, y)
		plt.show()
		plt.savefig(save_directory + str(i) +'-mixture-loss-samples.png')
		plt.close()

def scatter_plot_(X, num_components, file):
	x = np.arange(num_components+2)
	ys = [i+x+(i*x)**2 for i in range(num_components+2)]
	colors = cm.rainbow(np.linspace(0, 1, len(ys)))
	#print(colors)

	for n in range(num_components): 
		plt.scatter(X[n,:,0], X[n,:,1], color=colors[n], label=str(n))

	plt.scatter(X[num_components,:,0], X[num_components,:,1], color='b', label='7s')
	plt.scatter(X[num_components+1,:,0], X[num_components+1,:,1], color=colors[num_components+1], label='9s')
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig(file)
	plt.close()

def scatter_plot_100(X, num_components, file):
	x = np.arange(num_components)
	ys = [i+x+(i*x)**2 for i in range(num_components)]
	colors = cm.rainbow(np.linspace(0, 1, len(ys)))
	#print(colors)

	for n in range(num_components): 
		plt.scatter(X[n,0], X[n,1], color=colors[n], label=str(n))

	plt.legend(loc="upper left")
	plt.show()
	plt.savefig(file)
	plt.close()


def display_images(decoder, p_z, d, file, k = 50, data='mnist', b_data=None, b_mask=None):
	fig = plt.figure(figsize=(11, 7))
	# setting values to rows and column variables
	rows = 5
	columns = 10

	for i in range(k):
		fig.add_subplot(rows, columns, i+1)
		# showing image
		if data == 'mnist':
			x = generate_samples(p_z, decoder, d, L=1).cpu().data.numpy().reshape(1,1,28,28) 
			##So that we only impute the missing pixels
			if b_data is not None:
				a = b_data.cpu().data.numpy().reshape(1,1,28,28)  
				b = b_mask.cpu().data.numpy().reshape(1,1,28,28)   
				x[b] = a[b]
			plt.imshow(np.squeeze(x), cmap='gray', vmin=0, vmax=1)
		else:
			x = generate_samples(p_z, decoder, d, L=1, data = "svhn").cpu().data.numpy().reshape(3,32,32)  
			x = (255/2)*(1 + x)
			x = x.astype(int)
			x = np.transpose(x, (1, 2, 0))
			plt.imshow(x)
		plt.axis('off')
		#plt.title("missing image")
	plt.show()
	plt.savefig(file)
	plt.close()


def display_images_with_labels(decoder, means, scales, logits_y, d, file, k = 50, data='mnist', b_data=None, b_mask=None, directory2=None, file2=None,logits=None):
	fig = plt.figure(figsize=(11, 7))
	# setting values to rows and column variables
	rows = 5
	columns = 10
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	for i in range(k):
		fig.add_subplot(rows, columns, i+1)
		# showing image
		if data == 'mnist':
			x = generate_samples_with_labels( means, scales, logits_y, decoder, d, L=1, logits=logits).cpu().data.numpy().reshape(1,1,28,28) 
			#if b_data is not None:
			#a = b_data.cpu().data.numpy().reshape(1,1,28,28)  
			#b = b_mask.cpu().data.numpy().reshape(1,1,28,28)   
			#x[b] = a[b]
			plt.imshow(np.squeeze(x), cmap='gray', vmin=0, vmax=1)				
		else:
			x = generate_samples(p_z, decoder, d, L=1, data = "svhn").cpu().data.numpy().reshape(3,32,32)  
			x = (255/2)*(1 + x)
			x = x.astype(int)
			x = np.transpose(x, (1, 2, 0))
			plt.imshow(x)
		plt.axis('off')
		#plt.title("missing image")
	plt.show()
	plt.savefig(file)
	plt.close()

	if directory2 is not None:
		for comp in range(10):
			fig = plt.figure(figsize=(11, 7))
			rows = 5
			columns = 10
			for i in range(k):
				fig.add_subplot(rows, columns, i+1)
				labels = torch.zeros(1, 10)
				labels[0,comp] = 1
				if logits is None:
					q_zgiveny = td.Independent(td.Normal(loc=means[comp,:], scale=torch.nn.Softplus()(scales[comp,:])),1) 
				else:
					q_zgiveny = ReparameterizedNormalMixture1d(logits[comp].reshape(1,10), means[comp].reshape(1, 10, d), scales[comp].exp().reshape(1, 10, d))

				zgiveny = q_zgiveny.rsample([1])
				z_flat = zgiveny.reshape([1,d])
				y_flat = labels.reshape([1,10])
				zy_flat = torch.cat((z_flat,y_flat.to(device,dtype = torch.float)),1)

				#zgivenxy_flat = zy_flat.reshape([1,d+10])
				all_logits_obs_model = torch.sigmoid(decoder.forward(zy_flat))
				x = b_data
				x[~b_mask] = all_logits_obs_model[~b_mask]
				plt.imshow(np.squeeze(x.cpu().data.numpy().reshape(1,1,28,28) ), cmap='gray', vmin=0, vmax=1)
				plt.axis('off')
			plt.show()
			plt.savefig(directory2+str(comp)+file2)
			plt.close()



def display_images_svhn(decoder, p_z, d, file, k = 50):
	fig = plt.figure(figsize=(11, 7))

	# setting values to rows and column variables
	rows = 5
	columns = 10

	for i in range(k):
		fig.add_subplot(rows, columns, i+1)
		# showing image
		x = generate_samples(p_z, decoder, d, L=1, data = "svhn").cpu().data.numpy().reshape(3,32,32)  
		x = (255/2)*(1 + x)
		x = x.astype(int)
		x = np.transpose(x, (1, 2, 0))
		plt.imshow(x)
		plt.axis('off')
		#plt.title("missing image")

	plt.show()
	plt.savefig(file)
	plt.close()


def display_images_from_distribution(obs_x, mask, p, file, k = 50, data='mnist'):

	fig = plt.figure(figsize=(11, 7))
	# setting values to rows and column variables
	rows = 5
	columns = 10

	for i in range(k):
		fig.add_subplot(rows, columns, i+1)
		# showing image
		x = obs_x
		x[~mask] = p.sample().cpu().float()
		if data == 'mnist':
			plt.imshow(np.squeeze(x.data.numpy()), cmap='gray', vmin=0, vmax=1)
		else:
			x = x.data.numpy().reshape(3,32,32)   
			x = (255/2)*(1 + x)
			x = x.astype(int)
			x = np.transpose(x, (1, 2, 0))
			plt.imshow(x)
		plt.axis('off')
		#plt.title("missing image")
	plt.show()
	plt.savefig(file)
	plt.close()


def plot_image(img, file='true.png', missing_pattern_x = None, missing_pattern_y = None):

	#plt.subplot(121)
	#plt.imshow(np.squeeze(img))
	#plt.imshow(img)
	plt.imshow(img, cmap='gray', vmin=0, vmax=1)
	if missing_pattern_x is not None: 
		plt.scatter(missing_pattern_y, missing_pattern_x)

	#plt.ticklabel_format()
	plt.show()
	plt.savefig(file)
	plt.close()

def plot_image_svhn(img, file='true.png', missing_pattern_x = None, missing_pattern_y = None):

	#plt.subplot(121)
	#plt.imshow(np.squeeze(img))
	#plt.imshow(img)
	std = [0.2023, 0.1994, 0.2010]
	mean = [0.4914, 0.4822, 0.4465]

	#img[0,:,:] = img[0,:,:]*std[0] + mean[0]
	#img[1,:,:] = img[1,:,:]*std[1] + mean[1]
	#img[2,:,:] = img[2,:,:]*std[2] + mean[2]

	print(np.max(img), np.min(img))
	#img = (img)*(255)

	print(np.max(img), np.min(img))
	#exit()
	#img = img.astype(int)
	img = expit(img)
	img =  np.transpose(img, (1, 2, 0))
	plt.imshow(img)
	if missing_pattern_x is not None: 
		plt.scatter(missing_pattern_y, missing_pattern_x)

	#plt.ticklabel_format()
	plt.show()
	plt.savefig(file)
	plt.close()


def plot_all_averages(image1, image2, file):
	fig = plt.figure(figsize=(3, 3))

	# setting values to rows and column variables
	rows = 2
	columns = 1

	fig.add_subplot(rows, columns, 1)

	# showing image
	plt.imshow(image1)
	plt.axis('off')
	plt.title("Comparing m.s.e, ")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 2)
	
	# showing image
	plt.imshow(image2)
	plt.axis('off')
	plt.title("joint log-likelihood")

	plt.show()
	plt.savefig(file)
	plt.close()


def plot_images(image1,image2, image3, image4, image5, image6, image7, image8, image0, file):
	fig = plt.figure(figsize=(6, 5))

	# setting values to rows and column variables
	rows = 3
	columns = 3

	fig.add_subplot(rows, columns, 1)
	# showing image
	plt.imshow(image1)
	plt.axis('off')
	plt.title("missing image")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 2)
	# showing image
	plt.imshow(image2)
	plt.axis('off')
	plt.title("true image")

	fig.add_subplot(rows, columns, 3)
	# showing image
	plt.imshow(image3)
	plt.axis('off')
	plt.title("burn-in image")

	fig.add_subplot(rows, columns, 4)
	# showing image
	plt.imshow(image4)
	plt.axis('off')
	plt.title("init")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 5)
	# showing image
	plt.imshow(image5)
	plt.axis('off')
	plt.title("pseudo-gibbs ")

	fig.add_subplot(rows, columns, 6)
	# showing image
	plt.imshow(image6)
	plt.axis('off')
	plt.title("metropolis ")

	fig.add_subplot(rows, columns, 7)
	# showing image
	plt.imshow(image7)
	plt.axis('off')
	plt.title("our method (iter:9)")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 8)
	# showing image
	plt.imshow(image8)
	plt.axis('off')
	plt.title("our method (iter:49)")

	fig.add_subplot(rows, columns, 9)
	# showing image
	plt.imshow(image0)
	plt.axis('off')
	plt.title(" our method ")

	plt.show()
	plt.savefig(file)
	plt.close()


def plot_images_z(image1,image2, image3, file1, image4, image5, file_save, iters):
	fig = plt.figure(figsize=(6, 5))

	# setting values to rows and column variables
	rows = 4
	columns = 4

	fig.add_subplot(rows, columns, 1)
	# showing image
	plt.imshow(image1)
	plt.axis('off')
	plt.title("missing image")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 2)
	# showing image
	plt.imshow(image2)
	plt.axis('off')
	plt.title("true image")

	fig.add_subplot(rows, columns, 3)
	# showing image
	plt.imshow(image3)
	plt.axis('off')
	plt.title("init/burn-in")

	k = 0
	a = iters/100
	my_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	my_new_list = [int(i * a) for i in my_list]
	for _ in my_new_list:
		fig.add_subplot(rows, columns, 4 + k)
		# showing image
		img = plt.imread(file1 + str(_ -1) + "z-output.png")
		plt.imshow(img)
		plt.axis('off')
		plt.title("output iter " + str(_))
		k += 1 

	fig.add_subplot(rows, columns, 15)
	# showing image
	plt.imshow(image4)
	plt.axis('off')
	plt.title("pseudo-gibbs")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 16)
	# showing image
	plt.imshow(image5)
	plt.axis('off')
	plt.title("metropolis")

	plt.show()
	plt.savefig(file_save)
	plt.close()



def plot_gibbs_images(image1,image2, image3, image4, image5, image6, image7, image8, image0, file):
	fig = plt.figure(figsize=(6, 5))

	# setting values to rows and column variables
	rows = 3
	columns = 3

	fig.add_subplot(rows, columns, 1)
	# showing image
	plt.imshow(image1)
	plt.axis('off')
	plt.title("iter 1")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 2)
	# showing image
	plt.imshow(image2)
	plt.axis('off')
	plt.title("iter 2")

	fig.add_subplot(rows, columns, 3)
	# showing image
	plt.imshow(image3)
	plt.axis('off')
	plt.title("iter 3")

	fig.add_subplot(rows, columns, 4)
	# showing image
	plt.imshow(image4)
	plt.axis('off')
	plt.title("iter 4")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 5)
	# showing image
	plt.imshow(image5)
	plt.axis('off')
	plt.title("miter 5")

	fig.add_subplot(rows, columns, 6)
	# showing image
	plt.imshow(image6)
	plt.axis('off')
	plt.title("iter 6")

	fig.add_subplot(rows, columns, 7)
	# showing image
	plt.imshow(image7)
	plt.axis('off')
	plt.title("iter 7")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 8)
	# showing image
	plt.imshow(image8)
	plt.axis('off')
	plt.title("iter 8")

	fig.add_subplot(rows, columns, 9)
	# showing image
	plt.imshow(image0)
	plt.axis('off')
	plt.title("iter 9")

	plt.show()
	plt.savefig(file)
	plt.close()

def plot_5runs_helper(loss, num_images, colours, x, ylabel, save_directory, save_location, ylim1 = None, ylim2 = None ):

	for image in range(num_images):
		for k_iters in range(6):
		    #print(xm_loss[k_iters],colours[k_iters])
		    if k_iters==0:
		        plt.plot(x, loss[k_iters, image], color=colours[k_iters], label="true image")
		    else:
		        plt.plot(x, loss[k_iters, image], color=colours[k_iters], label="our method " + str(k_iters))

		plt.xlabel('Iterations')
		plt.ylabel(ylabel) 
		ylim1 = loss[0,image,-1] - 50
		ylim2 = loss[0,image,-1] + 20
		#if ylim1 is not None:
		plt.ylim(ylim1, ylim2)
		plt.legend(loc="upper left")
		plt.show()
		plt.savefig(save_directory + str(image) + '-' + save_location)
		plt.close()
	return

def compare_ELBO_helper(loss1, loss2, loss3, loss4, loss5, colours, x, ylabel, save_location, ylim1= None, ylim2 = None):

	plt.plot(x, loss1, color=colours[0], label="O_XM")
	#plt.plot(x, loss4, color=colours[3], label="O_XM_NN")
	plt.plot(x, loss2, color=colours[1], label="IAF")
	plt.plot(x, loss3, color=colours[2], label="O_Z")
	plt.plot(x, loss5, color=colours[4], label="Mixture")
	
	plt.xlabel('Iterations')
	plt.ylabel(ylabel) 
	if ylim1 is not None:
		plt.ylim(ylim1, ylim2)

	#plt.ylim(top=0)
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig(save_location)
	plt.close()

def compare_iwae(lower_bound, upper_bound, bound_updated_encoder, bound_updated_test_encoder, pseudo_gibbs_iwae, metropolis_within_gibbs_iwae, loss1, loss2, loss3, loss4, colours, x, ylabel, save_location, ylim1= None, ylim2 = None):

	x__ = np.arange(10)
	ys = [i+x__+(i*x__)**2 for i in range(10)]
	colours = cm.rainbow(np.linspace(0, 1, len(ys)))

	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ms = 12
	ax.plot(x, lower_bound, color='red', label="0's")
	ax.plot(x, upper_bound, color="green", label="True")
	ax.plot(x, bound_updated_encoder, color='orange',linestyle='--', label="0's + tuned encoder (train)")
	ax.plot(x, bound_updated_test_encoder, color=colours[0], label="0's + tuned encoder (test)")

	ax.plot(x, pseudo_gibbs_iwae, color="pink", linestyle='--', label="Pseudo Gibbs")
	ax.plot(x, metropolis_within_gibbs_iwae, color="purple", label="Metropolis Within Gibbs")

	ax.plot(x, loss1, color="yellow", label="Gaussian")
	ax.plot(x, loss2, color="brown", label="IAF")
	ax.plot(x, loss3, color="blue", linestyle='--',  label="Mixture")
	ax.plot(x, loss4, color="olive", label="Mixture (re-inits)")
	
	#ax.xlabel()
	#ax.ylabel() 

	if ylim1 is not None:
		ax.axis(ymin=ylim1,ymax=ylim2)
		#ax.ylim(ylim1, ylim2)

	#plt.ylim(top=0)
	#lgd = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
	#lgd = ax.legend(loc='lower center', ncol=5)
	lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	#ax.set_title('IWAE bounds vs #Samples')
	ax.set_xlabel('#Samples')
	ax.set_ylabel(ylabel)   

	#plt.show()
	#plt.savefig()
	fig.savefig(save_location, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

	#plt.close()

def plot_curve(y,num_epochs, file):
	x = np.arange(1, num_epochs + 1, 1).reshape(num_epochs)
	plt.plot(x, y)
	plt.show()
	plt.savefig(file)
	plt.close()

def plot_5runs(num_epochs, xm_loss, iaf_loss, z_loss, xm_loss_NN, mixture_loss, results, i, ylim1=None, ylim2=None):

	x = np.arange(1, num_epochs + 1, 1).reshape(num_epochs)
	colours = ['g', 'b', 'y', 'r', 'k', 'c']

	plot_5runs_helper(loss= xm_loss, num_images = 10, colours = colours, x = x, ylabel= 'Loss (NELBO)', save_directory = results + str(i) + "/compiled/", save_location = 'xm_loss.png' , ylim1 = ylim1, ylim2 = ylim2)
	plot_5runs_helper(loss= iaf_loss, num_images = 10, colours = colours, x = x, ylabel= 'Loss (NELBO)', save_directory = results + str(i) + "/compiled/", save_location = 'iaf-loss.png',ylim1 = ylim1, ylim2 = ylim2 )
	plot_5runs_helper(loss= z_loss, num_images = 10, colours = colours, x = x, ylabel= 'Loss (NELBO)', save_directory = results + str(i) + "/compiled/", save_location ='z-loss.png' ,ylim1 = ylim1, ylim2 = ylim2 )
	plot_5runs_helper(loss= xm_loss_NN, num_images = 10, colours = colours, x = x, ylabel= 'Loss (NELBO)', save_directory = results + str(i) + "/compiled/", save_location = 'xm_loss_NN.png', ylim1 = ylim1, ylim2 = ylim2 )
	plot_5runs_helper(loss= mixture_loss, num_images = 10, colours = colours, x = x, ylabel= 'Loss (NELBO)', save_directory = results + str(i) + "/compiled/", save_location = 'mixture_loss.png' , ylim1 = ylim1, ylim2 = ylim2)

	return

def compare_ELBO(num_epochs, xm_loss, iaf_loss, z_loss, xm_loss_NN, mixture_loss, results, i, ylim1= None, ylim2=None, image=0):
	x = np.arange(1, num_epochs + 1, 1).reshape(num_epochs)
	colours = ['g', 'b', 'y', 'r', 'k', 'c']
	num_images = 10

	for iter_ in range(6):
		compare_ELBO_helper(loss1 = xm_loss[iter_], loss2 = iaf_loss[iter_], loss3 = z_loss[iter_], loss4 = xm_loss_NN[iter_], loss5 = mixture_loss[iter_], colours = colours, x = x, ylabel= 'Loss (NELBO)', save_location = results + str(i) + "/compiled/" + str(iter_) + 'comparison-' + str(image) +'.png', ylim1 = ylim1, ylim2 = ylim2)

def plot_images_in_row(num_epochs, loc1, loc2, loc3, loc4, loc5, file, data='mnist'):
	image1 = plt.imread(loc1)
	image2 = plt.imread(loc2)
	image3 = plt.imread(loc3)
	image4 = plt.imread(loc4)
	image5 = plt.imread(loc5)

	fig = plt.figure(figsize=(4, 1))

	# setting values to rows and column variables
	rows = 1
	columns = 5

	#print(image1.shape)
	fig.add_subplot(rows, columns, 1)
	# showing image
	plt.imshow(image1)
	#plt.imshow(image1)
	plt.axis('off')
	plt.title("0 ")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 2)
	# showing image

	plt.imshow(image2)
	plt.axis('off')
	plt.title(str(int(num_epochs/4)  ))

	fig.add_subplot(rows, columns, 3)
	# showing image
	plt.imshow(image3)
	plt.axis('off')
	plt.title(str(int(2*num_epochs/4) ) )

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 4)
	# showing image
	plt.imshow(image4)
	plt.axis('off')
	plt.title(str(int(3*num_epochs/4) ) )

	fig.add_subplot(rows, columns, 5)
	# showing image
	plt.imshow(image5)
	plt.axis('off')
	plt.title(str(int(num_epochs) ) )

	plt.show()
	plt.savefig(file)
	plt.close()

def plot_images_comparing_methods(images, file, data='mnist'):
	fig = plt.figure(figsize=(4, 1))

	# setting values to rows and column variables
	rows = 1
	columns = 10

	for i in range(len(images)):
		fig.add_subplot(rows, columns, i+1)
		# showing image
		plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
		#plt.imshow(image1)
		plt.axis('off')
		#plt.title("0 ")

	plt.show()
	plt.savefig(file)
	plt.close()

def plot_labels_in_row(images, logqy,  file, data='mnist'):

	fig = plt.figure(figsize=(4, 1))

	# setting values to rows and column variables
	rows = 1
	columns = 10

	for i in range(10):
		fig.add_subplot(rows, columns, i+1)
		# showing image
		plt.imshow(images[i])
		#plt.imshow(image1)
		plt.axis('off')
		plt.title("q:" + str(logqy[0,i]))

	plt.show()
	plt.savefig(file)
	plt.close()

def plot_all_images(i, nb, iterations, file, prefix, data='mnist'):

	image1 = plt.imread(prefix + "image-inits.png")
	image2 = plt.imread(prefix + "pg-all.png")
	image3 = plt.imread(prefix + "mwg-all.png")
	image4 = plt.imread(prefix + "q_xm-all.png")
	image5 = plt.imread(prefix + "q_xm-all-NN.png")
	image6 = plt.imread(prefix + "iaf-all.png")
	image7 = plt.imread(prefix + "z-output-all.png")


	fig = plt.figure(figsize=(1, 2))

	# setting values to rows and column variables
	rows = 7
	columns = 1

	fig.add_subplot(rows, columns, 1)
	# showing image
	plt.imshow(image1)
	plt.axis('off')
	plt.title("(a)")

	fig.add_subplot(rows, columns, 2)
	# showing image
	plt.imshow(image2)
	plt.axis('off')
	plt.title("(b)")

	fig.add_subplot(rows, columns, 3)
	# showing image
	plt.imshow(image3)
	plt.axis('off')
	plt.title("(c)")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 4)
	# showing image
	plt.imshow(image4)
	plt.axis('off')
	plt.title("(d)")

	fig.add_subplot(rows, columns, 5)
	# showing image
	plt.imshow(image5)
	plt.axis('off')
	plt.title("(e)")

	fig.add_subplot(rows, columns, 6)
	# showing image
	plt.imshow(image6)
	plt.axis('off')
	plt.title("(f)")

	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 7)
	# showing image
	plt.imshow(image7)
	plt.axis('off')
	plt.title("(g)")

	plt.show()
	plt.savefig(file)
	plt.close()





