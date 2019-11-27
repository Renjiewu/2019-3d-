# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 18:32:21 2019

@author: DELL
"""

import numpy as np
import pandas as pd
import math
def find(p1):
#    p1=np.array(p1)
#    p1=np.mat(p1)
    #data=pd.read_csv("C:/Users/DELL/Desktop/8/8.csv")
    #data2=pd.read_csv("C:/Users/DELL/Desktop/8/4.csv")
    #data1=np.asanyarray(data)
    #data3=np.asanyarray(data2)
#    m=np.mat(np.zeros((1,4)))
#    for i in range(0,2):
#        m[0,i]=p1[0,i]
#    for i in range(0,2):
#        m[0,i+2]=p2[0,i]
#    for i in range(0,3):
#        m[0,i]=int(math.floor(m[0,i]))
    
    a=p1.shape[0]    #行
    b=p1.shape[1]


    n1=p1
    
#    for i in range(0,a):
#        for j in range(0,b):
#            n1[i,j]=data3[a1+i-1,b1+j]
    n2=np.mat(np.zeros((1,b)))   
    for j in range(0,b):
        for i in range(0,a):
            if n1[a-i-1,j]>0:
                continue
            else:
                n2[0,j]=a-i-1
                break
#    print(n2)
    n3=np.mat(np.zeros((1,b)))
    for j in range(0,b):
        for i in range(0,int(n2[0,j])):
            if n1[int(n2[0,j]-1-i),j]==0:
                continue
            else:
                n3[0,j]=int(n2[0,j]-1-i)
                break
    n4=np.mat(np.zeros((1,b)))
    for j in range(0,b):
        for i in range(0,int(n3[0,j])):
            if n1[int(n3[0,j]-1-i),j]>0:
                continue
            else:
                n4[0,j]=int(n3[0,j]-1-i)
                break         
    n5=np.mat(np.zeros((1,b)))
    for j in range(0,b):
        for i in range(0,int(n4[0,j])):
            if n1[int(n4[0,j]-1-i),j]==0:
                continue
            else:
                n5[0,j]=int(n4[0,j]-1-i)
                break
    #print(n5)
    n6=np.mat(np.zeros((1,b)))
    for j in range(0,b):
        for i in range(0,int(n5[0,j])):
            if n1[int(n5[0,j]-1-i),j]>n1[int(n5[0,j]-2-i),j]:
                continue
            else:
                if n1[int(n5[0,j]-1-i),j]<n1[int(n5[0,j]-2-i),j]:
                    n6[0,j]=int(n5[0,j]-1-i)
                    break
    n7=np.mat(np.zeros((1,b)))
    for j in range(0,b):
        for i in range(0,int(n6[0,j])):
            if n1[int(n6[0,j]-1-i),j]>0:
                continue
            else:
                if n1[int(n6[0,j]-1-i),j]==0:
                    n7[0,j]=n6[0,j]
                    break
    c=0
    for i in range(0,b):
        if n7[0,i]>0:
            c=c+1
            continue
    m1=np.mat(np.zeros((int(c),2)))
    d=0
    for i in range(0,b):
        if n7[0,i]>0:
            d=d+1
            m1[d-1,0]=n7[0,i]
            m1[d-1,1]=i
           
    return m1
#data=pd.read_csv("C:/Users/DELL/Desktop/8/8.csv")
#data2=pd.read_csv("C:/Users/DELL/Desktop/8/4.csv")
#data1=np.asanyarray(data)
#data3=np.asanyarray(data2)
#m=np.mat(np.zeros((1,4)))
##print(data1)
#for i in range(0,4):
#     m[0,i]=data1[0,i+1]
#
#for i in range(0,4):
#        m[0,i]=int(math.floor(m[0,i]))
#print(m)
#a=m[0,2]-m[0,0]+1
#b=m[0,3]-m[0,1]+1
#a1=m[0,1]
#b1=m[0,0]
#n1=np.mat(np.zeros((int(a),int(b)))) 
#for i in range(0,int(a)):
#        for j in range(0,int(b)):
#            n1[i,j]=data3[int(b1+i-1),int(a1+j)]
#print(n1[0,16])
#find(n1)