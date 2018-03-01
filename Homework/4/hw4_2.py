# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:38:37 2018

@author: Zhimiao
"""

import numpy as np
import math
from numpy import linalg as la
import matplotlib.pyplot as plt

def ISTA(H,y,lam,x):
    eig=la.eig(np.dot(np.transpose(H),H))
    alpha = np.max(eig[0])
#    x=[[0],[1],[0]]
    T=1e-7
    rnew=0.1
    thold=lam/(2*alpha)
    while True:
#    for k in range(0,100):
        rold = rnew
        temp=np.dot(np.transpose(H),y-np.dot(H,x))/alpha
#        for i in range(0,np.size(x)):    
        x = np.minimum(np.add(x,temp+thold),np.zeros(1))+np.maximum(np.add(x,temp-thold),np.zeros(1))
        rnew = la.norm(y-np.dot(H,x),2) + np.dot(lam,la.norm(x,1))
        ratio = np.abs((rold - rnew)/rold)
        if ratio < T:
            break
    return x

def DCT_basis_gen(N):
    h=[]
    for k in range(0,N):
        if k == 0:
            a=1/np.sqrt(N)
        else:
            a=np.sqrt(2)/np.sqrt(N)
        h.append([])
        for n in range(0,N):
            h[k].append(a*np.cos((2*n+1)*k*math.pi/(2*N)))
    return h

h=DCT_basis_gen(16)
#print(h)

x=np.zeros((16,1))
x[3]=40
x[8]=90
x[15]=20
SNRdb=10
SNR=10**(SNRdb/10)
w=np.random.normal(0,scale=np.max(x)/np.sqrt(SNR),size=np.shape(x))
y=np.dot(h,x)+w
lam=30
J=ISTA(h,y,lam,x)
print(J)
error=J-x
plt.plot(J,label='denoised signal')
plt.plot(error,label='error',color='r')
#plt.plot(w,label='noise',color='r')