# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:55:51 2018

@author: Zhimiao
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from numpy import linalg as la

    
def inverse_transform(x):
    return pywt.waverec2(x,'haar')
#    return pywt.waverec2(x,'db8')
    
def forward_transform(y,lev):
    return pywt.wavedec2(y,'haar',level=lev)
#    return pywt.wavedec2(y,'db8',level=lev)

def soft(x,thold):
    return np.minimum(np.add(x,thold),np.zeros(1))+np.maximum(np.add(x,-thold),np.zeros(1))
    
def ISTA_Orth(y,lam):
    wn=forward_transform(y,3)
    wn,_=pywt.coeffs_to_array(wn)
    
    LL,(LH,HL,HH)=forward_transform(y,1)
    LL=soft(LL,lam/2)
    LL2,(LH2,HL2,HH2)=forward_transform(LL,1)
    LL2=soft(LL2,lam/2)
    LL3,(LH3,HL3,HH3)=forward_transform(LL2,1)
    LL3=soft(LL3,lam/2)
    LH3=soft(LH3,lam/2)
    HL3=soft(HL3,lam/2)
    HH3=soft(HH3,lam/2)
    LH2=soft(LH2,lam/2)
    HL2=soft(HL2,lam/2)
    HH2=soft(HH2,lam/2)
    LH=soft(LH,lam/2)
    HL=soft(HL,lam/2)
    HH=soft(HH,lam/2)
    wdn=(LL3,(LH3,HL3,HH3),(LH2,HL2,HH2),(LH,HL,HH))
    
    final=inverse_transform(wdn)
    
    return wn,wdn,final


img=cv2.imread('Lena512gray.png',0)

# change SNR here:
SNRdb=10
SNR=10**(SNRdb/10)
w=np.random.normal(0,scale=np.max(img)/SNR,size=np.shape(img))
nimg=img+w

ximg=np.dot(nimg,0)
# change lamda here:
lam=40
wnimg,wdnimg,Fimg=ISTA_Orth(nimg,lam)

wdnimg=list(wdnimg)
wdnimg[0]=np.resize(wdnimg[0],(64,64))
for i in range(1,4):
    for j in range(0,3):
        wdnimg[i]=list(wdnimg[i])
        wdnimg[i][j]=np.resize(wdnimg[i][j],(64*(2**(i-1)),64*(2**(i-1))))
dnlev1=np.vstack((np.hstack((wdnimg[0],wdnimg[1][0])),np.hstack((wdnimg[1][1],wdnimg[1][2]))))
dnlev2=np.vstack((np.hstack((dnlev1,wdnimg[2][0])),np.hstack((wdnimg[2][1],wdnimg[2][2]))))
wdnimg=np.vstack((np.hstack((dnlev2,wdnimg[3][0])),np.hstack((wdnimg[3][1],wdnimg[3][2]))))

#wdnimg,_=pywt.coeffs_to_array(wnimg)

fig=plt.figure(figsize=(16,24))
ax1=plt.subplot(321)
#plt.figure(figsize=(12,12))
plt.imshow(img,cmap=plt.cm.gray)
ax1.get_xaxis().set_visible(False) 
ax1.get_yaxis().set_visible(False) 
ax1.set_title('original image')
ax2=plt.subplot(322)
#plt.figure(figsize=(12,12))
plt.imshow(nimg,cmap=plt.cm.gray)
ax2.get_xaxis().set_visible(False) 
ax2.get_yaxis().set_visible(False) 
ax2.set_title('noisy image')
ax3=plt.subplot(323)
#plt.figure(figsize=(12,12))
plt.imshow(wnimg,cmap=plt.cm.gray)
ax3.get_xaxis().set_visible(False) 
ax3.get_yaxis().set_visible(False) 
ax3.set_title('wavelet transform of noisy image')
ax4=plt.subplot(324)
#plt.figure(figsize=(12,12))
plt.imshow(wdnimg,cmap=plt.cm.gray)
ax4.get_xaxis().set_visible(False) 
ax4.get_yaxis().set_visible(False) 
ax4.set_title('wavelet transform of denoised image')
ax5=plt.subplot(325)
#plt.figure(figsize=(12,12))
plt.imshow(Fimg,cmap=plt.cm.gray)
ax5.get_xaxis().set_visible(False) 
ax5.get_yaxis().set_visible(False) 
ax5.set_title('final denoised image')
ax6=plt.subplot(326)
error=Fimg-img
#plt.figure(figsize=(12,12))
plt.imshow(error,cmap=plt.cm.gray)
ax6.get_xaxis().set_visible(False) 
ax6.get_yaxis().set_visible(False) 
ax6.set_title('error')
#plt.imsave('haar transform.jpg')
