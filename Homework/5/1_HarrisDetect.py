import cv2
import numpy as np
#import scipy as sp
from scipy.ndimage import filters
from scipy import signal
import matplotlib.pyplot as plt
#import math

#%%
def GaussianFilt(img,win,sigma):
    g=np.ones((win,win))
    d = np.int((win-1)/2)
    for x in range(-d,d+1):
        for y in range(-d,d+1):
            g[x+2,y+2]=np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    fimg=signal.convolve2d(img,g, boundary = 'symm')
    return fimg

#img=cv2.imread('BK_left.jpg',0)
#test = GaussianFilt(img,5,1)
##cv2.imshow('test',test)
#plt.figure(figsize=(12,12))
#plt.imshow(test,cmap=plt.cm.gray)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
    
#%%
def Gaussian_1d(win,sigma):    
    gx=np.ones((win,win))
    gy=np.ones((win,win))
    g=np.ones((win,win))
    d = np.int((win-1)/2)
    for x in range(-d,d+1):
        for y in range(-d,d+1):
            g[x+2,y+2]=np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
            gx[x+2,y+2]=-x*g[x+2,y+2]/(sigma**2)
            gy[x+2,y+2]=-y*g[x+2,y+2]/(sigma**2)
    return gx,gy

#%%
def Harris(img,sigma,win,swin):
    
    rows,cols = img.shape[:2]
#    img = cv2.cvtColor(simg,cv2.COLOR_BGR2GRAY)
    
    #Ix=filters.gaussian_filter(img,sigma,axis=0,order=1,truncate=win)
    #Iy=filters.gaussian_filter(img,sigma,axis=1,order=1,truncate=win)
    gx, gy=Gaussian_1d(win,sigma)
    Ix=signal.convolve2d(img,gx,boundary = 'symm')
    Iy=signal.convolve2d(img,gy,boundary = 'symm')
    #cv2.imshow('Ix',Ix)
    #cv2.imshow('Iy',Iy)
    #plt.figure(figsize=(12,12))
    #plt.imshow(Ix,cmap=plt.cm.gray)
    #plt.figure(figsize=(12,12))
    #plt.imshow(Iy,cmap=plt.cm.gray)
    
    # Ix^2,Iy^2,IxIy
    Ix2=Ix**2
    Iy2=Iy**2
    Ixy=Iy*Ix
    #cv2.imshow('Ix2',Ix2)
    #cv2.imshow('Iy2',Iy2)
    #cv2.imshow('IxIy',IxIy)
    #plt.figure(figsize=(12,12))
    #plt.imshow(Ix2,cmap=plt.cm.gray)
    #plt.figure(figsize=(12,12))
    #plt.imshow(Iy2,cmap=plt.cm.gray)
    #plt.figure(figsize=(12,12))
    #plt.imshow(Ixy,cmap=plt.cm.gray)
    
    # smooth each image
    
#    Ix2_s=filters.gaussian_filter(Ix2,2*sigma,truncate=swin)
#    Iy2_s=filters.gaussian_filter(Iy2,2*sigma,truncate=swin)
#    Ixy_s=filters.gaussian_filter(Ixy,2*sigma,truncate=swin)
    
    Ix2_s=GaussianFilt(Ix2,swin,2*sigma)
    Iy2_s=GaussianFilt(Iy2,swin,2*sigma)
    Ixy_s=GaussianFilt(Ixy,swin,2*sigma)
    #    t=GaussianFilt(img,1,2)
    #    cv2.imshow('test',t)
    #cv2.imshow('Ix2_s',Ix2_s)
    #cv2.imshow('Iy2_s',Iy2_s)
    #cv2.imshow('IxIy_s',IxIy_s)
    #cv2.waitKey(0)                      
    #cv2.destroyAllWindows()
    
    #%%
    #Harris cornerness value image
    
    H = np.zeros((rows,cols))
    for i in range(3,rows-3):
        for o in range(3,cols-3):
            a00 = np.sum(Ix2_s[i-2:i+3,o-2:o+3])
            a01 = np.sum(Ixy_s[i-2:i+3,o-2:o+3])
            a11 = np.sum(Iy2_s[i-2:i+3,o-2:o+3])
            H[i,o] = a00*a11-a01**2 - 0.06*(a00+a11)**2
    H[H<0] = 0
    H = H/np.max(H)*255
    
    #plt.figure(figsize=(12,12))
    #plt.imshow(H,cmap=plt.cm.gray)
#    cv2.imshow('H',H)
    
    #%%
    #H=cv2.cornerHarris(nimg3,2,9,0.06)
    #cv2.imshow('H',H)
    #print(H)
    #cv2.waitKey(0)                      
    #cv2.destroyAllWindows()
    #%%
    #    detect local maxima
    cft=[]
    
    for i in range(2,rows-2):
        for j in range(2,cols-2):
            local = np.array([H[i-2:i+3,j-2:j+3]])
            if np.max(local) == H[i,j] and np.max(local)!= 0:
                cft.append([H[i,j],(j,i)])

    N=50
    ft=sorted(cft,reverse=True)[0:N]
#    print(ft)

    for i in range(N):
        dimg=cv2.circle(img,ft[i][1],3,(255,255,255),-1)
    cv2.imshow('corner detecter',dimg)

#        cv2.imshow('image',img)
    
#    return dimg
    cv2.waitKey(0)                      
    cv2.destroyAllWindows()
    
    return ft
    
#%%
#img=cv2.imread('Berries.jpg',0)
img=cv2.imread('BK_left.jpg',0)

sigma=1
win=4*sigma+1
swin=6*sigma+1

features = Harris(img,sigma,win,swin)
    
#cv2.waitKey(0)                      
#cv2.destroyAllWindows()

#%%
# rotate and resize, create new images

img=cv2.imread('BK_left.jpg',0)
rows,cols = img.shape[:2]

nimg1=cv2.resize(img,(cols/2,rows/2))
nimg2=cv2.resize(img,(cols*2,rows*2))

M=cv2.getRotationMatrix2D(((cols/2,rows/2)),30,1)
nimg3=cv2.warpAffine(nimg1,M,(cols,rows))
M=cv2.getRotationMatrix2D(((cols/2,rows/2)),-20,1)
nimg4 = cv2.warpAffine(nimg2,M,(cols*2,rows*2))

#cv2.imshow('new1',nimg1)
#cv2.imshow('new2',nimg2)
#cv2.imshow('new3',nimg3)
cv2.imshow('new4',nimg4)
#
cv2.waitKey(0)
cv2.destroyAllWindows()

#himg3 = Harris(nimg3,sigma,win,swin)

ft4 = Harris(nimg4,sigma,win,swin)
