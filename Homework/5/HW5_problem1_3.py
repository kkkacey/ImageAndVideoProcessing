import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math


#%%

# Start of problem 1

def GaussianFilt(img,win,sigma):
    g=np.ones((win,win))
    d = np.int((win-1)/2)
    for x in range(-d,d+1):
        for y in range(-d,d+1):
            g[x+2,y+2]=np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    fimg=signal.convolve2d(img,g, boundary = 'symm')
    return fimg

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
    
    gx, gy=Gaussian_1d(win,sigma)
    Ix=signal.convolve2d(img,gx,boundary = 'symm')
    Iy=signal.convolve2d(img,gy,boundary = 'symm')
    
    Ix2=Ix**2
    Iy2=Iy**2
    Ixy=Iy*Ix
    
    Ix2_s=GaussianFilt(Ix2,swin,2*sigma)
    Iy2_s=GaussianFilt(Iy2,swin,2*sigma)
    Ixy_s=GaussianFilt(Ixy,swin,2*sigma)
    
    H = np.zeros((rows,cols))
    for i in range(3,rows-3):
        for o in range(3,cols-3):
            a00 = np.sum(Ix2_s[i-2:i+3,o-2:o+3])
            a01 = np.sum(Ixy_s[i-2:i+3,o-2:o+3])
            a11 = np.sum(Iy2_s[i-2:i+3,o-2:o+3])
            H[i,o] = a00*a11-a01**2 - 0.06*(a00+a11)**2
    H[H<0] = 0
    H = H/np.max(H)*255
    
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
    cv2.waitKey(0)                      
    cv2.destroyAllWindows()
    
    return ft, dimg

#%%
img=cv2.imread('BK_left.jpg',0)
rimg=cv2.imread('BK_right.jpg',0)

sigma=1
win=4*sigma+1
swin=6*sigma+1

ft,dimg = Harris(img,sigma,win,swin)

#%%
# create new images by rotating and resizing
rows,cols = img.shape[:2]

nimg1=cv2.resize(img,(cols/2,rows/2))
nimg2=cv2.resize(img,(cols*2,rows*2))

M=cv2.getRotationMatrix2D(((cols/2,rows/2)),30,1)
nimg3=cv2.warpAffine(nimg1,M,(cols,rows))
M=cv2.getRotationMatrix2D(((cols/2,rows/2)),-20,1)
nimg4 = cv2.warpAffine(nimg2,M,(cols*2,rows*2))

#cv2.imshow('new1',nimg1)
#cv2.imshow('new2',nimg2)
cv2.imshow('new image, downsize',nimg3)
cv2.imshow('new image, upsize',nimg4)
#
cv2.waitKey(0)
cv2.destroyAllWindows()

ft4, dimg4 = Harris(nimg4,sigma,win,swin)


# end of problem 1


#%%

# start of problem 2


def MagAndAngle(img,win,sigma,N):
    w,h = img.shape[:2]
    
    gx, gy=Gaussian_1d(win,sigma)
    Ix=signal.convolve2d(img,gx)
    Iy=signal.convolve2d(img,gy)
    
    ori = np.zeros((w,h))
    mag = np.zeros((w,h))
    for x in range(1,w-1):
        for y in range(1,h-1):
            l = Ix[x-1:x+2,y-1:y+2]
            mag[x,y] = np.sqrt( (l[2,1] - l[0,1])**2 + (l[1,2] - l[1,0])**2 )
            ori[x,y] = math.atan( (l[1,2] - l[1,0]) / (l[2,1] - l[0,1]) ) 
    
    #quantisize        
    N = 8
    q = 45
    ori_q = np.floor( (ori + q/2)/q )
    for i in range(1,w-1):
        for j in range(1,h-1):
            if ori_q[i,j] == N:
                x = 0
    return mag, ori_q
 #%%
def SIFT(img,featureP):
    w,h = img.shape[:2]
    win = 16
    N=8
    x,y = featureP[:][1]
    
    mag,ori = MagAndAngle(img,5,1,N)
    
    patchMag = mag[x-win/2 : 1+x+win/2 , y-win/2 : 1+y+win/2]
    w_patchMag = GaussianFilt(patchMag, 3, sigma = win/2)
    hog = [0]*N
    for i in range(w):
        for j in range(h):
            for p in range(N):
                if ori[i,j] == p:
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
                        if ori[i,j] == p:
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

#%%

def match(ft1,ft2,r):
#    ft1 = Harris(img1,1,5,7)
#    ft2 = Harris(img2,1,5,7)
    
#    ft1P = ft1[:][1]
#    ft2P = ft2[:][1]
    
    matchPairs = []
    sumft = 0
    for i in range(len(ft1)):
        dis = np.zeros(len(ft2))
        for i2 in range(len(ft2)):
            sumft = (ft1[i][1][0] - ft2[i2][1][0])**2 + (ft1[i][1][1] - ft2[i2][1][1])**2
            dis[i2] = sumft**0.5
        dis = list(dis)
        q1 = dis.index(min(dis))
        q1v = dis[q1]
        dis[q1] = max(dis)
        q2 = dis.index(min(dis))
        if q1v/dis[q2] < r:
            matchPairs.append([i,q1])
    
    return matchPairs

# end of problem 2



#%%
    

# start of problem 3
#    feature points and descriptors of original image already generated before.

M2=cv2.getRotationMatrix2D(((cols/2,rows/2)),2,1)
rimg=cv2.warpAffine(img,M2,(cols,rows))

ftr, rdimg = Harris(rimg,sigma,win,swin)

#%%
#show in one image
def showMatchingPairs(dimg1,dimg2,matchPairs):
    rows,cols = dimg1.shape[:2]
    rows2,cols2 = dimg2.shape[:2]
    bimg = np.zeros((rows, cols + cols2))
    bimg[0:rows,0:cols] = dimg1[:,:]
    bimg[0:rows,cols:cols + cols2] = dimg2[:,:]
    
    
    for i in range(np.shape(matchPairs)[0]):
        pt1 = ft[matchPairs[i][0]]
        pt2 = ftr[matchPairs[i][1]]
        pt2[1] = list(pt2[1])
    #    pt2[1][1] = pt2[1][1] + cols
        cv2.line(bimg, (pt1[1][0],pt1[1][1]), (pt2[1][0] + cols, pt2[1][1]), (255,255,255))
    #    print([pt1[1][1],pt1[1][0],[pt2[1][1], pt2[1][0]]])
    
    return bimg

matchPairs = match(ft,ftr,0.3)
#cv2.imshow('matching',bimg)
#cv2.waitKey(0)                      
#cv2.destroyAllWindows()

bimg = showMatchingPairs(dimg,rdimg,matchPairs)

plt.figure(figsize = (15,15))
plt.imshow(bimg, cmap = plt.cm.gray)


#  end of problem 3
