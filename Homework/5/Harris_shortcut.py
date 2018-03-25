import cv2
import numpy as np
#import scipy as sp
from scipy.ndimage import filters
from scipy import signal
import matplotlib.pyplot as plt

img=cv2.imread('BK_left.jpg',0)
#img=cv2.imread('checkerboard.png',0)

#sigma=1
#win=4*sigma+1
#Ix=filters.gaussian_filter1d(img,sigma,axis=0,order=1,truncate=win)
#Iy=filters.gaussian_filter1d(img,sigma,axis=1,order=1,truncate=win)
#
#Ix2=Ix**2
#Iy2=Iy**2
#IxIy=Iy*Ix
#
#swin=6*sigma+1
#Ix2_s=filters.gaussian_filter(Ix2,2*sigma,truncate=swin)
#Iy2_s=filters.gaussian_filter(Iy2,2*sigma,truncate=swin)
#IxIy_s=filters.gaussian_filter(IxIy,2*sigma,truncate=swin)

H=cv2.cornerHarris(img,2,7,0.06)
#cv2.imshow('H',H)
plt.figure()
plt.imshow(H,cmap=plt.cm.gray)
#print(H)

cft=[]
for i in range(1,np.shape(H)[0]):
    for j in range(1,np.shape(H)[1]):
        window = H[i-1:i+2,j-1:j+2].copy()
        window[1,1] = 255
        if H[i,j] < np.min(window): #and H[i,j] > 220:
            cft.append([H[i,j],(j,i)])
N=50
ft=sorted(cft,reverse=True)[0:N]
#print(ft)
#    
for i in range(len(ft)):
    dimg=cv2.circle(img,ft[i][1],5,(255,255,255))
cv2.imshow('corner detecter',dimg)
#plt.imshow(dimg)
#    cv2.imshow('image',img)

cv2.waitKey(0)                      
cv2.destroyAllWindows()