# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 08:32:09 2016

@author: aladwig
"""

import cv2
import numpy as np
from scipy.misc import imrotate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import cPickle as pickle

img_rows = 64
img_cols = 80
border_size = 2
use_existing_clf = True

def load_train_data():
    img_files = ['1.jpg','2.jpg']
    mask_files = ['1_mask.jpg', '2_mask.jpg']
    imgs = []
    masks = []
    for i in range(len(img_files)):
        img = cv2.imread(img_files[i],cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
    
        img = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        img_mask = cv2.resize(img_mask, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        
        imgs.append(img)
        masks.append(img_mask)
    imgs = np.array(imgs)
    masks = np.array(masks)

    p_imgs, p_masks = perturb_imgs(imgs, masks)

    return p_imgs, p_masks

def perturb_imgs(imgs, masks):
    return_images = []
    return_masks = []
    
    for idx in range(len(imgs)):
        img = imgs[idx]
        mask = masks[idx]
       
        for _ in range(100):
            #translate and rotate img by random amounts
            #masks get matching translation and rotations
            p_img = img
            p_mask = mask
            
            x_shift = np.random.randint(0, img_cols)
            y_shift = np.random.randint(0,img_rows)
            deg = np.random.randint(0,360)
    
            m1 = np.float32([[1,0,-x_shift*0.25], [0,1,-y_shift*0.25]])
    
            p_img = cv2.warpAffine(p_img,m1,(img_cols,img_rows))
            p_img = imrotate(p_img, deg)
            
            p_mask = cv2.warpAffine(p_mask,m1,(img_cols,img_rows))
            p_mask = imrotate(p_mask, deg)

            # Gather the border pixels for each pixel in p_img into a 3x3 array 
            # Border pixels get padded with zeros
            out = []
            for r in range(0,img_rows):
                for c in range(0,img_cols):
                    p = []
                    for i in range(-border_size,border_size+1):
                        for j in range(-border_size,border_size+1):
                            if((r+i)==-1 or (c+j)==-1):
                                p.append(0)
                            else:
                                try:
                                    p.append(p_img[r+i,c+j]) 
                                except IndexError:
                                    p.append(0)
                    
                    out.append(np.reshape(p,(-1)))
            return_images.append(np.array(out))
         
         
            #convert p_mask values to either 0 or 255
            for r in range(0,img_rows):
                for c in range(0,img_cols):
                    if(p_mask[r,c] < 128):
                        p_mask[r,c] = 0
                    else:
                        p_mask[r,c] = 255
            
            p_mask = np.reshape(p_mask,(-1))
            return_masks.append(p_mask)     
         
    return_images = np.array(return_images).reshape(-1,(border_size*2+1)**2)
    return_masks = np.array(return_masks).reshape(-1)
         
    return return_images, return_masks
    

   
def load_test_imgs():
    img = cv2.imread('3.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    cols = img.shape[1]
    rows = img.shape[0]
    
    # Gather the border pixels for each pixel in p_img into a 3x3 array 
    out = []
    for r in range(0,rows):
        for c in range(0,cols):
            p = []
            for i in range(-border_size,border_size+1):
                for j in range(-border_size,border_size+1):
                    if((r+i)==-1 or (c+j)==-1):
                        p.append(0)
                    else:
                        try:
                            p.append(img[r+i,c+j]) 
                        except IndexError:
                            p.append(0)
            
            out.append(np.reshape(p,(-1)))
   
    return(np.array(out))
    
def run():
    if use_existing_clf == False:
        imgs, img_masks = load_train_data()
        clf = KNeighborsClassifier(n_jobs=-1)
        clf.fit(imgs, img_masks)
    else:
        with open('clf.pickle', 'rb') as f:
            clf = pickle.load(f)
   
    
    test_img = load_test_imgs()
    pred = clf.predict(test_img)
    pred = np.array(pred).reshape(img_rows,img_cols)
    pred = cv2.resize(pred, (0,0), fx=5.0, fy=5.0)
    cv2.imshow('',pred)
    cv2.waitKey()
    
    if use_existing_clf ==False:
        with open('clf.pickle', 'wb') as f:
            pickle.dump(clf,f)


if __name__ == '__main__':
    run()
