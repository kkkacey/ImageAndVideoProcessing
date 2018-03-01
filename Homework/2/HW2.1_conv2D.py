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

    h_len=np.size(h,1)
    h_wid=np.size(h,1)
    img_len=int(np.size(img,0))
    img_wid=int(np.size(img,1))
    img=np.pad(img,(h_len-1,h_wid-1),'constant',constant_values=(0,0))
    y=np.zeros((img_len+h_len-1,img_wid+h_wid-1))
     
    hhh=(h_len-1)/2;
    hhw=(h_wid-1)/2;
    for m in range(hhh+1,img_len-hhh):
        for n in range(hhw+1,img_wid-hhw):
            y[m,n]=np.sum(np.sum(np.dot(img[m-hhh:m+hhh+1,n-hhw:n+hhw+1],h)))
           
    return y


def main():

    img=cv2.imread('lena512gray.png',0)
    
    h1=np.divide([[1,2,1],[2,4,2],[1,2,1]],16.)
    h2=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    h3=[[0,-1,0],[-1,5,-1],[0,-1,0]]
    
    Fimg1=conv2D(img,h1)
    Fimg2=conv2D(img,h2)
    Fimg3=conv2D(img,h3)
    
        
    fig=plt.figure(figsize=(16,16))
    ax1=plt.subplot(221)
    plt.imshow(img,cmap=plt.cm.gray)
    ax1.get_xaxis().set_visible(False) 
    ax1.get_yaxis().set_visible(False) 
    ax1.set_title('original image')
    ax2=plt.subplot(222)
    plt.imshow(Fimg1,cmap=plt.cm.gray)
    ax2.get_xaxis().set_visible(False) 
    ax2.get_yaxis().set_visible(False) 
    ax2.set_title('filtered image 1')
    ax3=plt.subplot(223)
    plt.imshow(Fimg2,cmap=plt.cm.gray)
    ax3.get_xaxis().set_visible(False) 
    ax3.get_yaxis().set_visible(False) 
    ax3.set_title('filtered image 2')
    ax4=plt.subplot(224)
    plt.imshow(Fimg3,cmap=plt.cm.gray)
    ax4.get_xaxis().set_visible(False) 
    ax4.get_yaxis().set_visible(False) 
    ax4.set_title('filtered image 3')
#    plt.imwrite('FilteredImages.png',fig)

    fig=plt.figure(figsize=(16,16))
    ax1=plt.subplot(221)
    fimg=np.log(np.fft.fftshift(np.fft.fft2(img))+1)
    plt.imshow(abs(fimg),cmap=plt.cm.gray)
    ax1.set_title('original image frequency response')
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
#    plt.imsave('FrequencyResponses.png',fig)
    
    
main()