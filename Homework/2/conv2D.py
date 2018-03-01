# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 15:34:09 2018

@author: Zhimiao
"""
import cv2
import numpy as np 
from numpy import fft
from scipy import signal
import matplotlib.pyplot as plt

def conv2D(img,h):

#    img_len=img.shape[0]
#    img_wid=img.shape[1]
    h_len=np.size(h,1)
    h_wid=np.size(h,1)
    img_len=int(np.size(img,0))
    img_wid=int(np.size(img,1))
    img=np.pad(img,(h_len-1,h_wid-1),'constant',constant_values=(0,0))
    y=np.zeros((img_len+h_len-1,img_wid+h_wid-1))
#    y=np.zeros((img_len,img_wid))
    
#    # convolution in rows
#    for i in range(0,img_len+h_len-1):
#        for j in range(0,h_len):
#            t=h[:,j]*img[:,i+h_len-j]
#            y[:,j]=y[:,j]+t
#    
#    # convolution in columns
#    for i in range(0,img_wid+h_wid-1):
#        for j in range(0,h_wid):
#            t=h[j,:]*img[i+h_wid-j,:]
#            y[j,:]=y[j,:]+t
     
    hhh=(h_len-1)/2;
    hhw=(h_wid-1)/2;
    for m in range(hhh+1,img_len-hhh):
        for n in range(hhw+1,img_wid-hhw):
            y[m,n]=np.sum(np.sum(np.dot(img[m-hhh:m+hhh+1,n-hhw:n+hhw+1],h)))
           
#    img_full=y[h_len/2:img_len+h_len/2,h_wid/2:img_wid+h_wid/2]
    
#    return img_full
    return y


def main():

    img=cv2.imread('lena512gray.png',0)
    
#    cv2.imshow('original image', img)
#    cv2.waitKey(0)                      
#    cv2.destroyAllWindows()
    
#    f=np.ones((3,3))/9
    h1=np.divide([[1,2,1],[2,4,2],[1,2,1]],16.)
    h2=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    h3=[[0,-1,0],[-1,5,-1],[0,-1,0]]
    
    Fimg1=conv2D(img,h1)
    Fimg2=conv2D(img,h2)
    Fimg3=conv2D(img,h3)
#    Fimg2=int(Fimg2*255/np.max(Fimg2))
#    Fimg0=signal.convolve2d(img,h3,mode='full')
#    Fimg=Fimg0*255/np.max(Fimg0)
    
#    return Fimg,Fimg0

#    cv2.imshow('filtered image',Fimg)
#    cv2.imshow(np.angle(Fimg), cmap='hsv')
    
#    hist,bins=np.histogram(Fimg2.flatten(),256,[0,256])
#    
#    cdf=hist.cumsum()
#    cdf_normalized=cdf*255/ cdf.max()
#    img_histeq=cdf_normalized[Fimg2]
    
    
#    plt.figure(figsize=(12,12))
#    plt.imshow(img,cmap=plt.cm.gray)
##    plt.get_xaxis().set_visible(False) 
##    plt.get_yaxis().set_visible(False) 
##    plt.set_title('original image')
#    plt.figure(figsize=(12,12))
#    plt.imshow(Fimg1,cmap=plt.cm.gray)
##    fig.get_xaxis().set_visible(False) 
##    fig.get_yaxis().set_visible(False) 
##    fig.set_title('filtered image 1')
#    plt.figure(figsize=(12,12))
#    plt.imshow(Fimg2,cmap=plt.cm.gray)
#    plt.figure(figsize=(12,12))
#    plt.imshow(Fimg3,cmap=plt.cm.gray)
##    plt.get_xaxis().set_visible(False) 
##    plt.get_yaxis().set_visible(False) 
##    plt.set_title('original image')
    
#    cv2.imshow('orignal image',img)
#    cv2.waitKey(0)                      
#    cv2.destroyAllWindows()
    
    fig=plt.figure(figsize=(16,16))
    ax1=plt.subplot(221)
    fimg=np.log(np.fft.fftshift(np.fft.fft2(img))+1)
#    fimg=
    plt.imshow(abs(fimg),cmap=plt.cm.gray)
    ax1.set_title('original image')
    ax2=plt.subplot(222)
    fimg1=np.fft.fftshift(np.fft.fft2(h1))
    fimg1=np.abs(np.fft.fftshift(np.fft.fft2(h1)))
    plt.imshow(abs(fimg1),cmap=plt.cm.gray)
    ax2.set_title('filter1')
    ax3=plt.subplot(223)
    fimg2=np.log(np.fft.fftshift(np.fft.fft2(h2)))
    plt.imshow(abs(fimg2),cmap=plt.cm.gray)
    ax3.set_title('filter2')
    ax4=plt.subplot(224)
    fimg3=np.log(np.fft.fftshift(np.fft.fft2(h3)))
    plt.imshow(abs(fimg3),cmap=plt.cm.gray)
    ax4.set_title('filter3')
    
    
main()