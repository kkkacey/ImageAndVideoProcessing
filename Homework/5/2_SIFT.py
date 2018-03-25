import cv2
import numpy as np
#import scipy as sp
from scipy.ndimage import filters
from scipy import signal
import matplotlib.pyplot as plt
import math

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
def GaussianFilt(img,win,sigma):
    g=np.ones((win,win))
    d = np.int((win-1)/2)
    for x in range(-d,d+1):
        for y in range(-d,d+1):
            g[x+2,y+2]=np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    fimg=signal.convolve2d(img,g, mode = 'same', boundary = 'symm')
    return fimg

#%%
# 1. Harris Detector
#    img=cv2.imread('Berries.jpg',0)
img=cv2.imread('BK_left.jpg',0)
#img=cv2.imread('checkerboard.png',0)
#cv2.imshow('image',img)
#cv2.waitKey(0)                      
#cv2.destroyAllWindows()

w,h = img.shape[:2]

#%% 
sigma=1
win=4*sigma+1
#Ix=filters.gaussian_filter1d(img,sigma,axis=0,order=1,truncate=win)
#Iy=filters.gaussian_filter1d(img,sigma,axis=1,order=1,truncate=win)
gx, gy=Gaussian_1d(win,sigma)
Ix=signal.convolve2d(img,gx)
Iy=signal.convolve2d(img,gy)
#cv2.imshow('Ix',Ix)
#cv2.imshow('Iy',Iy)

ori = np.zeros((w,h))
mag = np.zeros((w,h))
for x in range(1,w-1):
    for y in range(1,h-1):
        l = Ix[x-1:x+2,y-1:y+2]
        mag[x,y] = np.sqrt( (l[2,1] - l[0,1])**2 + (l[1,2] - l[1,0])**2 )
        ori[x,y] = math.atan( (l[1,2] - l[1,0]) / (l[2,1] - l[0,1]) )    

#%%
# quantisize orientations
N = 8
q = 45

ori_q = np.floor( (ori + q/2)/q )
#ori_q[ori_q == N] = 0
for i in range(1,w-1):
    for j in range(1,h-1):
        if ori_q[i,j] == N:
            x = 0
            
#%%

def SIFT(img,featureP):
    win = 16

    x,y = featureP[:][1]
    patchMag = mag[x-win/2 : 1+x+win/2 , y-win/2 : 1+y+win/2]
    w_patchMag = GaussianFilt(patchMag, 3, sigma = win/2)
    hog = [0]*N
    for i in range(w):
        for j in range(h):
            for p in range(N):
                if ori_q[i,j] == p:
                    hog[p] = hog[p] + w_patchMag[i][j]
    hog = list(hog)
    patchOri = hog.index(max(hog))
    
    hog44 = [[0]*4 for i in range(4)]
    w_patchMag44 = [[0]*4 for i in range(4)]
    patchOri44 = [[0]*4 for i in range(4)]
    hogs = []
    for m in range(4):
        for n in range(4):
            hog44[m][n] = [0]*N
            w_patchMag44[m][n] = w_patchMag[m:4*m,n:4*n]
            for i in range(w):
                for j in range(h):
                    for p in range(N):
                        if ori_q[i,j] == p:
                            hog44[m][n][p] = hog[m][n][p] + w_patchMag44[m][n][i][j]   
            hog44[m][n] = list(hog44[m][n])
            patchOri44[m][n] = hog44[m][n].index(max(hog[m][n]))
            hog44[m][n] = hog44[m][n][hog44[m][n].index(max(hog[m][n]))::] + hog44[m][n][:hog44[m][n].index(max(hog[m][n])):]
            hogs = hogs + hog44[m][n]
#    normalize
    hogs_n = np.linalg.norm(hogs, ord = 2)
    hogs_n[hogs_n > 0.2] = 0.2
    hogs_rn = np.linalg.norm(hogs_n, ord = 2)
    
    return hog, hogs_rn

