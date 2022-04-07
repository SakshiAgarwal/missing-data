# missing-data

Code for missing data at test time. 

1. mnist-resnet.py: This file trains on (complete) MNIST and tests on missing MNIST data with the baseline method of filling mixing pixel values with 0s.
2. mnist-xm.py: This file loads the trained model on complete MNIST and tests on missing MNIST data with the 'proposed method'. 
3. train.py: Conatins a function to train the resnet VAE architecture for MNIST
4. loss.py: contains the loss and imputation method for both baseline and 'proposed method'.
5. resnets.py: Gabe's implementation of resnet for VAEs.
6. data.py: contains functions to load train, validation and test set for MNIST data. Test set includes a certain % of missing pixel values. 
