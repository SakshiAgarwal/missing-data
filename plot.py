import matplotlib.pyplot as plt
import numpy as np
from data import *
#import cv2

def plot_image(img, file='true.png'):

	#plt.subplot(121)
	#plt.imshow(np.squeeze(img))
	plt.imshow(img)
	plt.show()
	plt.savefig(file)


def compare_images(imgs, file="comparison1.png"):


	return
