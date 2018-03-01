#Homework 1_2
#Histogram Equalization by cumsum()

"""
@author: Zhimiao
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('wiki.jpg',0)

hist,bins=np.histogram(img.flatten(),256,[0,256])

cdf=hist.cumsum()
cdf_normalized=cdf*255/ cdf.max()
img_histeq=cdf_normalized[img]

# plot histograms and images
fig=plt.figure(figsize=(15,7))
ax1=plt.subplot(121)
plt.hist(img.ravel(),256,[0,256])
ax1.set_title('original histogram')
ax2=plt.subplot(122)
plt.hist(img_histeq.ravel(),256,[0,256])
ax2.set_title('resulting histogram')
plt.show()

fig=plt.figure(figsize=(16,7))
ax3=plt.subplot(121)
plt.imshow(img,cmap=plt.cm.gray)
ax3.get_xaxis().set_visible(False) 
ax3.get_yaxis().set_visible(False) 
ax3.set_title('original image')
ax4=plt.subplot(122)
plt.imshow(img_histeq,cmap=plt.cm.gray) 
ax4.get_xaxis().set_visible(False) 
ax4.get_yaxis().set_visible(False) 
ax4.set_title('resulting image')
plt.show()

# histogram equalization by cv2
#equ = cv2.equalizeHist(img)
#res = np.hstack((img,equ)) 
#cv2.imshow('res.png',res)
#cv2.waitKey(0)                      
#cv2.destroyAllWindows()