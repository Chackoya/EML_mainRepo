#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:40:33 2020

@author: gama
"""
import cv2 
import os
def teste(imgPath):
    #gray = cv2.imread("StockImg/img5.png",cv2.IMREAD_GRAYSCALE)

    myList = os.listdir(imgPath)
    
    print(myList)
    
    for f in myList:
        print(f)
        path=imgPath+"/"+f
        prepro(path)
    

    


def prepro(imgPath):
   
    gray = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    #resize img and invert it (black background)
    gray = cv2.resize(255-gray , (28,28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #save the processed img
    #cv2.imwrite("StockImg/pro_img9.png",gray)
    #cv2.imwrite(imgPath[:-4]+"_pro"+imgPath[-4:],gray)

    flatten=gray.flatten()/255.0
    print("GG")
    """
    prediction = model.predict(flatten.reshape(1, 28, 28, 1))
    print()
    print(prediction)
    print(">>THE PREDICTION FOR YOUR IMAGE IS:",np.argmax(prediction))
    """
    
    


import numpy as np
def zzz ():
    
    l = np.zeros((5))
    print(l)
    
    string="[ "
    for elem in range(len(l)):
        if elem==len(l)-1:
            string= string+str(l[elem])
        else:
            
            string=string+str(l[elem])+" ; "
        
    string = string+" ]"
    print(string)
#teste("StockImg")

zzz()