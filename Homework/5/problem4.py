import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
img=cv2.imread('BK_left.jpg',0)
rimg=cv2.imread('BK_right.jpg',0)

sigma=1
win=4*sigma+1
swin=6*sigma+1

rows,cols = img.shape[:2]
#ft,dimg = Harris(img,sigma,win,swin)

#%%
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

#%%

# start of problem 4


detector = cv2.xfeatures2d.SIFT_create()
descriptor = cv2.xfeatures2d.SURF_create()

skpl = detector.detect(img)
skpl, sdl = descriptor.compute(img, skpl)
skpr = detector.detect(rimg)
skpr, sdr = descriptor.compute(rimg, skpr)


#%%
# mark feature points
for i in range(len(skpl)):
    limg_d = cv2.circle(img, (int(skpl[i].pt[1]),int(skpl[i].pt[0])), int(skpl[i].size), (255,255,255),-1)
for i in range(len(skpr)):
    rimg_d = cv2.circle(rimg, (int(skpl[i].pt[1]),int(skpl[i].pt[0])), int(skpr[i].size), (255,255,255),-1)
cv2.imshow('features in left',limg_d)
cv2.imshow('features in right',rimg_d)
cv2.waitKey(0)                      
cv2.destroyAllWindows()

#%%
def matchSkp(skp1,skp2,r):
    matchPairs = []
    sumft = 0
    for i in range(len(skp1)):
        dis = np.zeros(len(skp2))
        for i2 in range(len(skp2)):
#            sumft = (ft1[i][1][0] - ft2[i2][1][0])**2 + (ft1[i][1][1] - ft2[i2][1][1])**2
            sumft = (skp1[i].pt[0]- skp2[i2].pt[0])**2 + (skp1[i].pt[1]- skp2[i2].pt[1])**2
            dis[i2] = sumft**0.5
        dis = list(dis)
        q1 = dis.index(min(dis))
        q1v = dis[q1]
        dis[q1] = max(dis)
        q2 = dis.index(min(dis))
        if q1v/dis[q2] < r:
            matchPairs.append([i,q1])
    
    return matchPairs

matchPs = matchSkp(skpl,skpr,0.3)
#print(matchPs)

#%%
# (e)
srcPoints = [0]*len(matchPs)
dstPoints = [0]*len(matchPs)
for i in range(len(matchPs)):
    srcPoints[i] = (int(skpl[matchPs[i][0]].pt[0]),int(skpl[matchPs[i][0]].pt[1]))
    dstPoints[i] = (int(skpl[matchPs[i][1]].pt[0]),int(skpl[matchPs[i][1]].pt[1]))
hom = cv2.findHomography(np.asarray(srcPoints), np.asarray(dstPoints), cv2.RANSAC)

#%%
show = showMatchingPairs(sdl,sdr,matchPs)
plt.figure(figsize = (15,15))
plt.imshow(show, cmap = plt.cm.gray)

#%%
hom = list(hom)
M=cv2.getRotationMatrix2D(((cols/2,rows/2)),hom.index(max(hom)),1)

panar = np.zeros((rows+100,cols+100))

limg_trans = cv2.warpPerspective(img, M)
panar[0:,0:] = limg_trans[:,:]
w,h = limg_trans[:2]
panar[:rows+100,:cols+100] = rimg

plt.figure(figsize = (15,15))
plt.imshow(panar, cmap = plt.cm.gray)