# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:58:18 2019

@author: w3321
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from . import rs1_cv as rs1
import math
import time
import os
from . import square_distance_angle as da
import pandas as pd
from .. import output2 as op
from .. import find

pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()
imgk=np.ones([550,550,3],np.uint8)*255

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)


# Start streaming


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
global vtx

def GetLineLength(p1,p2):
    '''计算边长'''
    length = math.pow((p2[0]-p1[0]),2) + math.pow((p2[1]-p1[1]),2)+ math.pow((p2[2]-p1[2]),2)
    length = math.sqrt(length)
    return length

depth_pixel = [320, 240]
def draw_circle(event,x,y,flags,param):
    global depth_pixel
    if event == cv2.EVENT_MOUSEMOVE:
        depth_pixel = [x, y]
        
def get_deep():
    while True:    
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        break
    return depth_frame,color_frame

def get_xyz(p):
    global vtx
    i=p[1]*640+p[0]
    p1=[np.float(vtx[i][0]),np.float(vtx[i][1]),np.float(vtx[i][2])]
    return p1

def get_one(path1):
    global vtx
    #time.sleep(5)
    start=time.time()
    
    pipe_profile=pipeline.start(config)
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset,1)
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset,1)
    sensor = pipe_profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.enable_auto_exposure,1)
    sensor.set_option(rs.option.backlight_compensation,0.000)
    sensor.set_option(rs.option.brightness,0.000)
    sensor.set_option(rs.option.contrast,50.000)
    sensor.set_option(rs.option.gain,61.000)
    sensor.set_option(rs.option.gamma,220.000)
    sensor.set_option(rs.option.hue,0.000)
    sensor.set_option(rs.option.saturation,68.000)
    sensor.set_option(rs.option.sharpness,50.000)
    
    cv2.namedWindow('depth_frame')
    cv2.setMouseCallback('depth_frame',draw_circle)
    while True:
        depth_frame1,color_frame1=get_deep()
        depth_frame2,color_frame2=get_deep()
        depth_frame,color_frame=get_deep()
        img_color = np.asanyarray(color_frame.get_data())
        img_test =img_color.copy()
        img_depth1 = np.asanyarray(depth_frame1.get_data())
        img_depth2 = np.asanyarray(depth_frame2.get_data())
        img_depth = np.asanyarray(depth_frame.get_data())
        img_depthx=(img_depth1+img_depth2+img_depth)//3
        depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(img_depth1, alpha=0.03), cv2.COLORMAP_BONE)
        depth_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(img_depth2, alpha=0.03), cv2.COLORMAP_BONE)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_BONE)
        p,imgf=rs1.rs1_cv(depth_colormap1,depth_colormap2,depth_colormap)
        
        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile) 
        
        # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        # Map depth to color
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
        
        color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
        color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
        #print ('depth: ',color_point)
        #print ('depth: ',color_pixel)
        
        x1,y1=depth_pixel
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        vtx = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        tex = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        i = y1*640+x1
        '''
        i0=p[0][1]*640+p[0][0]
        i1=p[1][1]*640+p[1][0]
        i2=p[2][1]*640+p[2][0]
        i3=p[3][1]*640+p[3][0]
        
        p0=[np.float(vtx[i0][0]),np.float(vtx[i0][1]),np.float(vtx[i0][2])]#左下
        p1=[np.float(vtx[i1][0]),np.float(vtx[i1][1]),np.float(vtx[i1][2])]#右下
        p2=[np.float(vtx[i2][0]),np.float(vtx[i2][1]),np.float(vtx[i2][2])]#左上
        p3=[np.float(vtx[i3][0]),np.float(vtx[i3][1]),np.float(vtx[i3][2])]#右上
        '''
        
        p0=get_xyz(p[0])
        p1=get_xyz(p[1])
        p2=get_xyz(p[2])
        p3=get_xyz(p[3])
        
        l0=GetLineLength(p0,p1)
        l1=GetLineLength(p0,p2)
        l2=GetLineLength(p1,p3)
        if l0==0:
            continue
        
        r=550/l0#mm/xlength
        l1_1=int(r*l1)#左边像素长度
        l2_1=int(r*l2)#右边像素长度
        #print(r,p0,p1,p2,p3)
        pts1 = np.float32([[0,550],[550,550],[0,550-l1_1],[550,550-l2_1]])
        pts2 = np.float32([[p[0][0],p[0][1]],[p[1][0],p[1][1]],[p[2][0],p[2][1]],[p[3][0],p[3][1]]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(imgk,M,(640,480))
        dst=cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        imgn, cnt, h = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnt:       
            cnt=cnt[0]
            cv2.drawContours(img_color, [cnt], 0, (0,255,0), 1)
            approx = cv2.approxPolyDP(cnt,8,True)
            cv2.drawContours(img_color, [approx], 0, (0,0,255), 3)
            a=np.asarray(approx)
            a=a.reshape(-1,2)
            #print(tuple(a[0]))
            '''
            cv2.circle(img_color,tuple(a[0]),5,(255,0,0),-1)
            cv2.circle(img_color,tuple(a[1]),5,(255,0,0),-1)
            cv2.circle(img_color,tuple(a[2]),5,(255,0,0),-1)
            cv2.circle(img_color,tuple(a[3]),5,(255,0,0),-1)
            '''
            
        
        #print ('depth: ',[np.float(vtx[i][0]),np.float(vtx[i][1]),np.float(vtx[i][2])])
        cv2.circle(img_color,(x1,y1), 8, [255,0,255], thickness=-1)
        cv2.putText(img_color,"Dis:"+str(img_depth[y1,x1]), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"X:"+str(np.float(vtx[i][0])), (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"Y:"+str(np.float(vtx[i][1])), (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.putText(img_color,"Z:"+str(np.float(vtx[i][2])), (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,255])
        cv2.imshow('depth_frame',img_color)
        #cv2.imshow('3',depth_colormap)
        cv2.imshow('4',imgf)
        #cv2.imshow('5',dst)
        key = cv2.waitKey(1)
        end=time.time()
        if (end-start)>5:
            #data = pd.DataFrame(img_depth)
            #data.to_csv('./1.csv',index = False)
        #if key & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    pipeline.stop()
    cv2.imwrite(path1+'/test_image/img_test.jpg',img_test)
    return r,p0,p1,p2,p3,img_depth,img_test

def find_p(imgs):
    m=find.find(imgs)
    n=[]
    b=0
    for i in m:
        if imgs[int(i[0,0]),int(i[0,1])] !=0:
            #print([int(i[0,0]),int(i[0,1])])
            n+=[int(i[0,0]),int(i[0,1])]
    if n==[]:
        b=1
    n=np.asarray(n).reshape(-1,2)
    return n,b

def fake_det(img,pm0):
    p=0
    if (img[pm0[1]+30,pm0[0]]!=0):
        if abs(int(img[pm0[1]+30,pm0[0]])-int(img[pm0[1],pm0[0]]))<700:
                if abs(int(img[pm0[1],pm0[0]])-int(img[pm0[1]+30,pm0[0]]))>350:
                    #print(abs(int(img[pm0[1],pm0[0]])-int(img[pm0[1]+30,pm0[0]])))
                    print('error')
                    p=1
    return p

def out(list,p0,p1,p2,p3,img):
    z=[['ZA001',0,0,0,0],
       ['ZA002',0,0,0,0],
       ['ZA003',0,0,0,0],
       ['ZA004',0,0,0,0],
       ['ZA005',0,0,0,0],
       ['ZB001',0,0,0,0],
       ['ZB002',0,0,0,0],
       ['ZB003',0,0,0,0],
       ['ZB004',0,0,0,0],
       ['ZB005',0,0,0,0],
       ['ZB006',0,0,0,0],
       ['ZB007',0,0,0,0],
       ['ZB008',0,0,0,0],
       ['ZB009',0,0,0,0],
       ['ZB010',0,0,0,0],
       ['ZC001',0,0,0,0],
       ['ZC002',0,0,0,0],
       ['ZC003',0,0,0,0],
       ['ZC004',0,0,0,0],
       ['ZC005',0,0,0,0],
       ['ZC006',0,0,0,0],
       ['ZC007',0,0,0,0],
       ['ZC008',0,0,0,0],
       ['ZC009',0,0,0,0],
       ['ZC010',0,0,0,0],
       ['ZC011',0,0,0,0],
       ['ZC012',0,0,0,0],
       ['ZC013',0,0,0,0],
       ['ZC014',0,0,0,0],
       ['ZC015',0,0,0,0],
       ['ZC016',0,0,0,0],
       ['ZC017',0,0,0,0],
       ['ZC018',0,0,0,0],
       ['ZC019',0,0,0,0],
       ['ZC020',0,0,0,0],
       ['ZC021',0,0,0,0],
       ['ZC022',0,0,0,0],
       ['ZC023',0,0,0,0],
       ['CA001',0,0,0,0],
       ['CA002',0,0,0,0],
       ['CA003',0,0,0,0],
       ['CA004',0,0,0,0],
       ['CD001',0,0,0,0],
       ['CD002',0,0,0,0],
       ['CD003',0,0,0,0],
       ['CD004',0,0,0,0],
       ['CD005',0,0,0,0],
       ['CD006',0,0,0,0]]
    
    b=[['ZA001',1,0,15],
       ['ZA002',0,0,42],
       ['ZA003',1,1,20],
       ['ZA004',0,1,25],
       ['ZA005',1,0,35],
       ['ZB001',0,0,32],
       ['ZB002',0,0,32],
       ['ZB003',0,0,20],
       ['ZB004',1,0,15],
       ['ZB005',0,0,25],
       ['ZB006',0,0,40],
       ['ZB007',1,0,20],
       ['ZB008',0,0,32],
       ['ZB009',0,1,30],
       ['ZB010',1,0,20],
       ['ZC001',0,1,15],
       ['ZC002',0,1,15],
       ['ZC003',0,1,15],
       ['ZC004',0,1,15],
       ['ZC005',0,0,15],
       ['ZC006',0,1,15],
       ['ZC007',0,1,15],
       ['ZC008',0,1,15],
       ['ZC009',0,1,15],
       ['ZC010',0,1,15],
       ['ZC011',1,1,15],
       ['ZC012',1,1,15],
       ['ZC013',0,1,15],
       ['ZC014',0,0,15],
       ['ZC015',0,0,32],
       ['ZC016',0,0,32],
       ['ZC017',0,0,32],
       ['ZC018',0,0,32],
       ['ZC019',0,0,32],
       ['ZC020',0,0,32],
       ['ZC021',0,0,32],
       ['ZC022',0,0,32],
       ['ZC023',0,0,32],
       ['CA001',0,0,32],
       ['CA002',0,0,32],
       ['CA003',0,0,32],
       ['CA004',0,0,32],
       ['CD001',0,0,32],
       ['CD002',0,0,32],
       ['CD003',0,0,32],
       ['CD004',0,0,32],
       ['CD005',0,0,32],
       ['CD006',0,0,32]]
    for i in list[1:]:
        zr=0.
        y1=int(i[1])
        x1=int(i[2])
        y2=int(i[3])
        x2=int(i[4])
        x=(x1+x2)//2
        y=(y1+y2)//2
        p=[x,y]
        p10=[x-5,y]
        p11=[x+5,y]
        if b[int(i[6]-1)][2]==1:
            imgs=img[y1:y2,x1:x2]
            m,c=find_p(imgs)
            if c==0:
                #print(m)
                j=m.shape[0]//2
            
                pm0=[m[j][1]+x1,m[j][0]+y1]
                '''
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.03), cv2.COLORMAP_BONE)
                for l in m:
                    cv2.circle(depth_colormap,(int(l[1])+x1,int(l[0])+y1), 1, [0,0,255], thickness=-1)
                #img_color=cv2.imread('../../vision_output/output/result/img_test.jpg')
                #cv2.circle(img_color,(x,y), 8, [255,0,255], thickness=-1)
                cv2.circle(depth_colormap,(pm0[0],pm0[1]), 2, [255,0,255], thickness=-1)
                cv2.circle(depth_colormap,(pm0[0],pm0[1]+10), 2, [255,0,255], thickness=-1)
                cv2.imshow('1',depth_colormap)
                cv2.waitKey()
                cv2.destroyAllWindows()
                '''
            else:
                print('s')
                pm0=p
        else:
            pm0=p
        fd=fake_det(img,pm0)
        if fd ==1:
            continue
        pm=get_xyz(pm0)
        p21=get_xyz(p10)
        p22=get_xyz(p11)
        zb=da.finddistan(p0,p2,p1,pm)
        if b[int(i[6]-1)][1]==1:
            zr=da.findangle(p0,p2,p1,p21,p22)
            if (b[int(i[6]-1)][0]=='ZC011') or (b[int(i[6]-1)][0]=='ZC011'):
                pass
            elif img[p10[1],p10[0]]<img[p11[1],p11[0]]:
                zr=180-zr
        z[int(i[6]-1)][4]+=1
        z[int(i[6]-1)][1]+=zb[0]
        z[int(i[6]-1)][2]+=zb[1]-b[int(i[6]-1)][3]
        z[int(i[6]-1)][3]+=zr
        
    return z

'''  
path=os.path.dirname(os.path.abspath(__file__))
path1=os.path.normpath(path+'../../../vision_output/output')
start1=time.time()
r,p0,p1,p2,p3,img_d,img_c=get_one(path1)
op.tf_dete(pth=path1)

#pipeline.stop()
#print(r,p0,p1,p2,p3)
#print(path1)
#os.system('python '+path1)


data=pd.read_csv(path1+'/result/img_test.jpg.csv',header=None)
list=data.values.tolist()
z=out(list,p0,p1,p2,p3,img_d)
zc=np.asanyarray(z).reshape(-1,4)
print(zc)

end1=time.time()
print(end1-start1)
'''