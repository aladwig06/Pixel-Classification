# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:48:08 2016

@author: aladwig
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imrotate


img = np.load('imgs_mask_test.npy')

img = img[0].reshape((64,80))
img = cv2.resize(img, (0,0), fx=5.0, fy=5.0)


cv2.imshow('img',img)
cv2.waitKey()

#img = cv2.imread("1.jpg")
#mask = cv2.imread("1_mask.jpg")
#
#imgs = []
#masks = []
#for i in range(100):
#    cols = img.shape[1]
#    rows = img.shape[0]
#    
#    x = np.random.randint(0,cols)
#    y = np.random.randint(0,rows)
#    deg = np.random.randint(0,360)
#    
#    m1 = np.float32([[1,0,-x*0.25],[0,1,-y*0.25]])
#    
#    img1 = cv2.warpAffine(img,m1,(cols,rows))
#    img1 = imrotate(img1, deg)
#    
#    mask1 = cv2.warpAffine(mask,m1, (cols,rows))
#    mask1 = imrotate(mask1, deg)
#    
#    imgs.append(img1)
#    masks.append(mask1)
#
#
#    
##    cv2.imshow('',img1)
##    cv2.waitKey()
##    cv2.imshow('',mask1)
##    cv2.waitKey()
#print(np.array(imgs).shape)
    
