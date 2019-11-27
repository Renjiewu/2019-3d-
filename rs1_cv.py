# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:16:26 2019

@author: w3321
"""
import cv2
import numpy as np

def find_p(roi):
    l1=(0,100)
    r1=(640,100)
    l=[]
    r=[]
    im2, cnt, h = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnt:
        l += tuple(cnt[cnt[:,:,0].argmin()][0])
        r += tuple(cnt[cnt[:,:,0].argmax()][0])
    l2=np.asarray(l).reshape(-1,2)
    r2=np.asarray(r).reshape(-1,2)
    l1=tuple(l2[l2[:,0].argmin()])
    r1=tuple(r2[r2[:,0].argmax()])
    #print(l1)
    return l1,r1

def rs1_cv(img1,img2,img):
    k = np.ones((11,11),np.uint8)
    k2 = np.ones((11,11),np.uint8)
    k3 = np.ones((5,5),np.uint8)

    #img=cv.imread('./1.jpg')
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1,thr1 = cv2.threshold(img1,10,255,cv2.THRESH_BINARY)
    ret2,thr2 = cv2.threshold(img2,10,255,cv2.THRESH_BINARY)
    ret,thr = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
    imgb = cv2.bitwise_or(thr1,cv2.bitwise_or(thr2,thr))
    imgc = cv2.morphologyEx(imgb,cv2.MORPH_CLOSE,k3)
    imgc = cv2.morphologyEx(imgc,cv2.MORPH_OPEN,k)
    imgc = cv2.morphologyEx(imgc,cv2.MORPH_CLOSE,k2)

    roi=imgc[100:400,0:300]
    l,r=find_p(roi)
    l1=(l[0],l[1]+100)
    roia=img[l1[1]-8:l1[1]+8,l1[0]-8:l1[0]+8]
    l,r=find_p(roia)
    l3=(l[0]+l1[0]-8,l[1]+l1[1]-8)
    
    roi1=imgc[100:400,340:640]
    l,r=find_p(roi1)
    r1=(340+r[0],r[1]+100)
    roi2=img[r1[1]-8:r1[1]+8,r1[0]-8:r1[0]+8]
    l,r=find_p(roi2)
    r3=(r[0]+r1[0]-8,r[1]+r1[1]-8)
    
    roi3=imgc[l1[1]-200:l1[1]-75,l1[0]:l1[0]+100]
    l,r=find_p(roi3)
    l2=(l1[0]+l[0],l[1]+l1[1]-200)
    roi2=img[l2[1]-8:l2[1]+8,l2[0]-8:l2[0]+8]
    l,r=find_p(roi2)
    l4=(l[0]+l2[0]-8,l[1]+l2[1]-8)
    
    roi4=imgc[r1[1]-200:r1[1]-75,r1[0]-100:r1[0]]
    l,r=find_p(roi4)
    r2=(r1[0]-100+r[0],r[1]+r1[1]-200)
    roi2=img[r2[1]-8:r2[1]+8,r2[0]-8:r2[0]+8]
    l,r=find_p(roi2)
    r4=(r[0]+r2[0]-8,r[1]+r2[1]-8)
    
    imgf=cv2.cvtColor(imgb,cv2.COLOR_GRAY2BGR)
    cv2.circle(imgf,r3,5,(0,0,255),-1)
    cv2.circle(imgf,l3,5,(0,0,255),-1)
    cv2.circle(imgf,r4,5,(0,0,255),-1)
    cv2.circle(imgf,l4,5,(0,0,255),-1)
    
    p=(l3,r3,l4,r4)
    #print(p)
    cv2.imshow('1',imgc)
    #cv2.imshow('2',roia)
    cv2.waitKey(1)
    return p,imgf