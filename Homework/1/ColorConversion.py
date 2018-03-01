#HomeWork 1_1
#Find and display cetain color range of an image

"""
@author: Zhimiao
"""

import cv2
import numpy as np 

# load image
img=cv2.imread('colors.jpg',1)
# concert to HSV
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# define range of blue
blue_lower=np.array([110,50,50])
blue_upper=np.array([130,255,255])

# find the pixels in blue range in the HSV image
mask=cv2.inRange(img_hsv,blue_lower,blue_upper)

# create an image containing only the blue parts
img_blue=cv2.bitwise_and(img,img,mask=mask)

# display the images
cv2.imshow('original color image', img)	
cv2.imshow('mask image',mask)
cv2.imshow('segmented image',img_blue)

print('Switch to image view. Then press any key to close')

cv2.waitKey(0)                      
cv2.destroyAllWindows()

cv2.imwrite('colors_blue.jpg', img_blue)

