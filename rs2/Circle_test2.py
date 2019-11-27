# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:58:31 2019

@author: DELL
"""
import math
import numpy as np
from scipy.linalg import solve 
#A，B，C圆面上任意三点，D物体圆心
def find_distance(A,B,C,D):

    A=np.mat(A)
    B=np.mat(B)
    C=np.mat(C)
    D=np.mat(D)
    #方向向量
    a_1=B[0,0]-A[0,0]
    b_1=B[0,1]-A[0,1]
    c_1=B[0,2]-A[0,2]
    #方向向量
    a_2=C[0,0]-A[0,0]
    b_2=C[0,1]-A[0,1]
    c_2=C[0,2]-A[0,2]
    #法向量
    A_a=b_1*c_2-b_2*c_1
    B_b=a_2*c_1-a_1*c_2
    C_c=a_1*b_2-a_2*b_1
    #    (xo-x1)²+(xo-x1)²+(xo-x1)²=(xo-x1)²+(xo-x1)²+(xo-x1)²
    #    (xo-x1)²+(xo-x1)²+(xo-x1)²=(xo-x1)²=(xo-x1)²=(xo-x1)²
    #     约束：A_ax0+B_by0+C_cz0=....
    a_2=2*B[0,0]-2*A[0,0]
    b_2=2*B[0,1]-2*A[0,1]
    c_2=2*B[0,2]-2*A[0,2]
    a_3=2*B[0,0]-2*C[0,0]
    b_3=2*B[0,1]-2*C[0,1]
    c_3=2*B[0,2]-2*C[0,2]
    a=[[a_2,b_2,c_2],[a_3,b_3,c_3],[A_a,B_b,C_c]]
    #print(a)
    b_1=pow(B[0,0],2)-pow(A[0,0],2)+pow(B[0,1],2)-pow(A[0,1],2)+pow(B[0,2],2)-pow(A[0,2],2)
    b_2=pow(B[0,0],2)-pow(C[0,0],2)+pow(B[0,1],2)-pow(C[0,1],2)+pow(B[0,2],2)-pow(C[0,2],2)
    b_3=A_a*A[0,0]+B_b*A[0,1]+C_c*A[0,2]
    c=[b_1,b_2,b_3]
    x=solve(a,c)
    #print(x)
    A_1=A-x
    l=math.sqrt(pow(A_1[0,0],2)+pow(A_1[0,1],2)+pow(A_1[0,2],2))
    r=300/l
    #print(r)
    D_1=D-x
    d_d_1=math.sqrt(pow(D_1[0,0],2)+pow(D_1[0,1],2)+pow(D_1[0,2],2))
    d_d_2=d_d_1*r
    #print(d_d_2)
    d_d_3=(A_a*D_1[0,0]+B_b*D_1[0,1]+B_b*D_1[0,2])/math.sqrt(pow(A_a,2)+pow(B_b,2)+pow(C_c,2))*r
    d_d_4=abs(d_d_3)
    #print(d_d_4)
    d=math.sqrt(abs(pow(d_d_2,2)-pow(d_d_4,2)))
    #print(d)
    return d

#find_distance([-0.25935065746307373,0.040360745042562485,0.7069244980812073],[0.2934683561325073,0.03464334085583687,0.6634291410446167],[-0.2053905725479126,0.10701954364776611,0.6264330744743347], D=[-0.0714,-0.1747,0.747])