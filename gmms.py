import torch
import numpy as np
from plot import *
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import pickle


def train_gaussian_mixture(train_loader, encoder, d, batch_size, results, file_save, data='mnist', with_labels=False):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if data=='mnist':
		embeddings = np.zeros((60000, 2*d))
	else :
		embeddings = np.zeros((73257, 2*d))

	nb=0
	if with_labels:
		embeddings = np.zeros((10, 6742, 2*d))
		nb = np.zeros(10)
	#embeddings = []
	print(batch_size)
	for data in train_loader:
		b_data, b_mask, b_full, labels_one_hot  = data
		#labels = torch.argmax(labels_one_hot, dim=1).item()
		b_full = b_full.to(device,dtype = torch.float)

		out_encoder = encoder.forward(b_full)

		if not with_labels:
			embeddings[ nb : nb + b_data.shape[0], :] = out_encoder.cpu().data.numpy().astype(float)
			nb += b_data.shape[0]
		else:
			print(labels)
			embeddings[int(labels), int(nb[labels]), :] = out_encoder.cpu().data.numpy().astype(float)
			nb[labels] += 1

	if not with_labels:
		embeddings = embeddings[~np.all(embeddings == 0, axis=1)]
		print(len(embeddings))
		gm = GaussianMixture(n_components=100, covariance_type = 'diag', init_params = 'k-means++').fit(embeddings[:,:d])
		print(gm.weights_, gm.means_.shape, gm.covariances_.shape)
		X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(gm.means_)
		scatter_plot_100(X.reshape([100,2]), 100, results + str(-1) + "/" + str(with_labels) + "mixture-100-components.png")
		with open(file_save, 'wb') as file:
			pickle.dump(gm, file)
	else:
		means = np.zeros((10, 10, 50))
		scales = np.zeros((10, 10, 50))
		weights = np.zeros((10, 10))

		for i in range(10):
			embeddings_ = embeddings[i, ~np.all(embeddings[i] == 0, axis=1)] 
			print(len(embeddings_), embeddings_.shape)
			gm = GaussianMixture(n_components=10, covariance_type = 'diag', init_params = 'k-means++').fit(embeddings_[:,:d])
			means[i] = gm.means_
			scales[i] = np.sqrt(gm.covariances_)
			weights[i] = gm.weights_

		with open(file_save, 'wb') as file:
			pickle.dump([means,scales,weights], file)

               








