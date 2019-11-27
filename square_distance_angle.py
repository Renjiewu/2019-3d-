# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:04:48 2019

@author: DELL
"""
import math
import numpy as np
#i_1_1左下；i_2_2左上；i_3_3右下；randomMatrix_1中心点坐标；
def finddistan(i_1_1,i_2_2,i_3_3,randomMatrix_1):
    #randomMatrix=np.random.randint(MINNUM,MAXNUM,(ROW,COL))
    i_1=np.array(i_1_1)
    i_2=np.array(i_2_2)
    i_3=np.array(i_3_3)
    i_4=i_2-i_1           #方向向量
    i_5=i_3-i_1            #方向向量
    q_q=np.mat(i_4)
    w_w=np.mat(i_5)
    O_1=np.mat(i_1)
    O_2=np.mat(i_2)
    O_3=np.mat(i_3)
    m_1=O_1[0,0]-O_2[0,0]
    m_2=O_1[0,1]-O_2[0,1]
    m_3=O_1[0,2]-O_2[0,2]
    #求出原点坐标
    d_d=math.sqrt(pow(m_1,2)+pow(m_2,2)+pow(m_3,2))            #两点之间距离
    #print(d_d)
    l=math.sqrt(pow(w_w[0,0],2)+pow(w_w[0,1],2)+pow(w_w[0,2],2))
    r=550/l
    #print(r)
    m=np.mat(np.zeros((1,3))) 
    for i in range(0,3):
        m[0,i]=(55/(d_d*r))*q_q[0,i]+O_1[0,i]
    #print(m)
    a_1=q_q[0,0]
    b_1=q_q[0,1]
    c_1=q_q[0,2]    #a_1 b_1 c_1为一组方向向量
    a_2=w_w[0,0]
    b_2=w_w[0,1]
    c_2=w_w[0,2]    #a_2 b_2 c_2为一组方向向量
    A=b_1*c_2-b_2*c_1
    B=a_2*c_1-a_1*c_2
    C=a_1*b_2-a_2*b_1   #[A B C]为平面法向量 
    randomMatrix=np.array(randomMatrix_1)
    #print(randomMatrix)
    n=randomMatrix-m
    #print(n)
    n_1=O_1-m
    n_2=O_3-m  
    qw=n[0,0]-n_1[0,0]      #x1-x0
    we=n[0,1]-n_1[0,1]      #y1-y0
    er=n[0,2]-n_1[0,2]      #z1-z0
    t=(a_1*qw+b_1*we+c_1*er)/(pow(a_1,2)+pow(b_1,2)+pow(c_1,2))
    f=pow((a_1*t-qw),2)+pow((b_1*t-we),2)+pow((c_1*t-er),2)
    F=math.sqrt(f)*r
    #print(F)
    h_h=A*n[0,0]+B*n[0,1]+C*n[0,2]
    d=abs(h_h)/math.sqrt(pow(A,2)+pow(B,2)+pow(C,2))
    D=d*r
    delta_d=math.sqrt(pow(F,2)-pow(D,2))
    delta_d=int(delta_d)
    #print(delta_d)     #x
    qw=n[0,0]-n_2[0,0]      #x1-x0
    we=n[0,1]-n_2[0,1]     #y1-y0
    er=n[0,2]-n_2[0,2]      #z1-z0
    t=(a_2*qw+b_2*we+c_2*er)/(pow(a_2,2)+pow(b_2,2)+pow(c_2,2))
    f=pow((a_2*t-qw),2)+pow((b_2*t-we),2)+pow((c_2*t-er),2)
    sqrt_f=math.sqrt(f)*r
    delta_d_d=math.sqrt(pow(sqrt_f,2)-pow(D,2))
    deltaD=550-delta_d_d
    deltaD=int(deltaD)
    #print(deltaD)          #y
    zb=[delta_d,deltaD]
    return zb
#i_1_1左下；i_2_2左上；i_3_3右下 ;D_1物体上两点（左）；E_1物体上两点（右）
def findangle(i_1_1,i_2_2,i_3_3,D_1,E_1):
    i_1=np.array(i_1_1)
    i_2=np.array(i_2_2)
    i_3=np.array(i_3_3)
    D=np.array(D_1)
    E=np.array(E_1)
    O_1=np.mat(i_1)
    O_2=np.mat(i_2)
    D=np.mat(D)
    E=np.mat(E)
    i_4=i_2-i_1           #方向向量
    i_5=i_3-i_1            #方向向量
    q_q=np.mat(i_4)
    w_w=np.mat(i_5)
    m_1=O_1[0,0]-O_2[0,0]
    m_2=O_1[0,1]-O_2[0,1]
    m_3=O_1[0,2]-O_2[0,2]
    l=math.sqrt(pow(w_w[0,0],2)+pow(w_w[0,1],2)+pow(w_w[0,2],2))
    #求出原点坐标
    d_d=math.sqrt(pow(m_1,2)+pow(m_2,2)+pow(m_3,2))            #两点之间距离
    m=np.mat(np.zeros((1,3))) 
    for i in range(0,3):
        m[0,i]=(l/d_d)*q_q[0,i]+O_1[0,i]
    D=D-m
    E=E-m
    a_1=q_q[0,0]
    b_1=q_q[0,1]
    c_1=q_q[0,2]    #a_1 b_1 c_1为一组方向向量
    a_2=w_w[0,0]
    b_2=w_w[0,1]
    c_2=w_w[0,2]    #a_2 b_2 c_2为一组方向向量
    A=b_1*c_2-b_2*c_1
    B=a_2*c_1-a_1*c_2
    C=a_1*b_2-a_2*b_1   #[A B C]为平面法向量 
    t=(-A*D[0,0]-B*D[0,1]-C*D[0,2])/(pow(A,2)+pow(B,2)+pow(C,2))
    D_1=np.mat(np.zeros((1,3))) 
    for i in range(0,3):
        D_1[0,i]=A*t+D[0,i]
    #print(D_1)
    t_t=(-A*E[0,0]-B*E[0,1]-C*E[0,2])/(pow(A,2)+pow(B,2)+pow(C,2))
    E_1=np.mat(np.zeros((1,3)))
    for i in range(0,3):
        E_1[0,i]=A*t_t+E[0,i]
    #print(E_1)
    de=E_1-D_1
    #print(de)
    x_1=de[0,0]*w_w[0,0]+de[0,1]*w_w[0,1]+de[0,2]*w_w[0,2]
    x_2=(math.sqrt(pow(de[0,0],2)+pow(de[0,1],2)+pow(de[0,2],2)))*(math.sqrt(pow(w_w[0,0],2)+pow(w_w[0,1],2)+pow(w_w[0,2],2)))
    #print(x_1,x_2)
    x=x_1/x_2
    y_1=math.acos(x)
    y=(y_1/(math.pi))*180
    #print(y)
    return y
#  距离
'''
finddistan([-0.272001,0.175772,0.56594],
        [-0.2715056,0.1166030,0.6456810],
        [0.271134,0.154697,0.522694],
        [-0.0269,-0.1095,0.6691],
        )
# 角度
findangle([-0.27119114995,0.170454904437,0.575938463211],
          [-0.2797571123,0.109151624143,0.6576797366],
          [0.267309725,0.152876287699,0.5251938701],
          [-0.11142,0.0172,0.7021],
          [-0.0413,-0.00817,0.7321])
'''