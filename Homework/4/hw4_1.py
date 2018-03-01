# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:20:50 2018

@author: Zhimiao
"""

import numpy as np
from numpy import linalg as la
#import math

def ISTA(H,y,lam):
    eig=la.eig(np.dot(np.transpose(H),H))
    alpha = np.max(eig[0])
    x=[[0],[1],[0]]
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
    
# test
H=[[1/np.sqrt(2),1,0],[1/np.sqrt(2),0,1]]
y=[[2],[2]]
lam=3
x=ISTA(H,y,lam)
print(x)
