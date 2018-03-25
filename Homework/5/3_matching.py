import cv2
import numpy as np
#import scipy as sp
from scipy.ndimage import filters
from scipy import signal
import matplotlib.pyplot as plt
import math
from PIL import Image

#%%

img=cv2.imread('BK_left.jpg',0)
rows,cols = img.shape[:2]
M=cv2.getRotationMatrix2D(((cols/2,rows/2)),2,1)
rimg=cv2.warpAffine(img,M,(cols,rows))

#%%

def match(img1,img2,r):
    ft1 = Harris(img1,1,5,7)
    ft2 = Harris(img2,1,5,7)
    
    ft1P = ft1[:][1]
    ft2P = ft2[:][1]
    
    matchPairs = []
    
    for i in range(np.shape(ft1P)[0]):
        dis = np.zeros(np.shape(ft2P)[0])
        for i2 in range(np.shape(ft2P)[0]):
            dis[i2] = np.linalg.norm(ft1P[i] - ft2P[i2], ord = 2)
        dis = list(dis)
        q1 = dis.index(min(dis))
        dis[q1]=0
        q2 = dis.index(min(dis))
        if dis[q1]/dis[q2] < r:
            matchPairs.append([i,q1])
    
    return matchPairs

#%%

bimg = Image.new('RGB', (cols * 2, rows), 0xffffff) 
bimg.paste(img, (0,0,rows,cols))
bimg.paste(rimg,(rows,0,rows*2,cols)) 
#
#bimg=cat(2,img,rimg)

cv2.imshow('bimg',bimg)
cv2.waitKey(0)
cv2.destroyAllWindows()           
