# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:50:28 2017

@author: SONY
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import glob, os
flist = []
deslist = []
label = ["13","14","15","16","23","24","25","26","33","34","43","44","45","46","46-24","53","54","55","56"]
ar = np.empty((0,128))
i = 0
"""
for root, dirs , files in os.walk('./Image'):
    for d in dirs:
        
        for root1,dirs1, files1 in os.walk('./Image/'+d):
            for file in files1:
                if(file.endswith('.npy')):
                    deslist.append(file)
                
"""
for p in pathlib.Path('./descripteursift').iterdir():
    if p.is_dir():
        "print(p)"
        flist.append(p)
for t in flist:       
    for p in pathlib.Path(t).iterdir():
        if p.is_file():
           "print(p)"
           ar =np.concatenate((ar,np.load(p)),axis=0)
    np.save(label[i],ar)
    ar = np.empty((0,128))
    i+=1       
          
           
#print(len(deslist))
"""
for p in pathlib.Path('./Image').iterdir():
    if p.is_dir():
        "print(p)"
        flist.append(p)
for t in flist:       
    for p in pathlib.Path(t).iterdir():
        if p.endswith('.npy'):
            deslist.append(p)
"""     
           

