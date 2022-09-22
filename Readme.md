# missing-data

The main files are : 

1. patches_all.py  : missing-data inference on the simple VAE trained on MNIST, supports inference with q(x_m), variations in q(z) techniques. 
2. labels_patches_all.py  : missing-data inference on the M2 VAE model trained on MNIST, supports inference with q(z) (Gaussian and mixtures only) techniques.
3. svhn_patches_all.py  : missing-data inference on the simple VAE trained on SVHN, supports inference with q(x_m), variations in q(z) techniques. 
4. inference.py : The main .py files (ex patches_all.py etc) calls function from this file for the different inference techniques.
5. loss.py : The inference.py file calls functions from this file containing the different loss functions for inference.

Instructions to run mnist with top-halg missing: 

1. cd into the folder you will be running the scripts from. 
2. mkdir results
3. mkdir results/mnist-False--1
4. mkdir results/mnist-False--1/pickled_files/
5. mkdir results/mnist-False--1/compiled/
6. mkdir results/mnist-False--1/images/
7. python3 mnist-top-half.py #With gaussian prior
8. python3 mnist-top-half-mixtures.py #With mixture prior

At the end of completion, the code saves parameters for the different methods as a pickled file in results/mnist-False--1/pickled_files/ 
The code also saves the loss curves for the different methods in the same file. I have commented out parts of  code which saved imputations of images, but I have asked you to create certain folders nonetheless.