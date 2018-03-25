#from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import cv2
import numpy as np
import scipy as sp
from scipy.ndimage import filters
from scipy import signal

def GaussianFilt(img,win,sigma):
    g=np.ones((win,win))
    for x in range(0,win):
        for y in range(0,win):
            g[x,y]=np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    fimg=signal.convolve2d(img,g,boundary='symm')
    compare=filters.gaussian_filter(img,sigma,truncate=win)
    return fimg,compare

img=cv2.imread('BK_left.jpg',0)
sigma=1
win=5   #4*sigma+1
#
fimg,compare=GaussianFilt(img,win,sigma)
cv2.imshow('fimg',fimg)
#cv2.imshow('compare',compare)

def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

g=gaussian_kernel(win)
gimg=signal.convolve2d(img,g)
cv2.imshow('gimg from kernel',gimg)






cv2.waitKey(0)
cv2.destroyAllWindows()