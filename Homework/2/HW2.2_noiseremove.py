# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 19:42:58 2018

@author: Zhimiao
"""

import cv2
import numpy as np 
import scipy
import matplotlib.pyplot as plt

img=cv2.imread('lena512gray.png',0)

mean=0
#sigma=0.1
sigma=0.01
noise=255*np.random.normal(mean,sigma,(img.shape))
nimg=img+noise

#plt.figure(figsize=(12,12))
#plt.imshow(nimg,cmap=plt.cm.gray)

fig=plt.figure(figsize=(16,16))
ax1=plt.subplot(221)
plt.imshow(img,cmap=plt.cm.gray)
ax1.get_xaxis().set_visible(False) 
ax1.get_yaxis().set_visible(False) 
ax1.set_title('original image')
ax2=plt.subplot(222)
plt.imshow(nimg,cmap=plt.cm.gray)
ax2.get_xaxis().set_visible(False) 
ax2.get_yaxis().set_visible(False) 
ax2.set_title('noise added image, sigma={}'.format(sigma))

#average filter
#h1=np.divide([[1,2,1],[2,4,2],[1,2,1]],16.)
#h1=np.divide([[1,1,1],[2,4,2],[1,2,1]],16.)
h1=np.divide(np.ones((9,9)),81)
dnimg=conv2D(nimg,h1)
ax3=plt.subplot(223)
plt.imshow(dnimg,cmap=plt.cm.gray)
ax3.get_xaxis().set_visible(False) 
ax3.get_yaxis().set_visible(False) 
ax3.set_title('denoised by average filter, size=9')

#Gaussian gilter
dnimg2=scipy.ndimage.filters.gaussian_filter(nimg,1)
ax4=plt.subplot(224)
plt.imshow(dnimg2,cmap=plt.cm.gray)
ax4.get_xaxis().set_visible(False) 
ax4.get_yaxis().set_visible(False) 
ax4.set_title('denoised by Gaussian filter')

