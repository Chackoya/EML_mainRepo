#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file loads the pretrained models and test them.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow import keras
import cv2
import os
import csv
#GENERAL LOAD MODEL
def loadModel(modelName , imgToClassify = None,resultsFile=None,modeCSV=None):
    
    ###LOAD THE MODEL:
    loaded_model= keras.models.load_model(modelName)
    
    #less preprocessing (obj is most to take raw data to generalize)
    print("LOADING AND CLASSIFYING LOAD MODEL TF")
    if imgToClassify != None:
        print("Classifying your images...")
        #preprocessImg_Classify(loaded_model,imgToClassify)
        check_results(modelName,loaded_model,imgToClassify, resultsFile,modeCSV)
        print("Job finished.")
    
    
def check_results(model_name, loaded_model,imgsPath,resultsFilePath,modeCSV):
    if(not(os.path.isdir(imgsPath))): #If the input is a simple file (a single img) 
        pred_list = classifyImg_general(loaded_model,imgsPath)
        if resultsFilePath!=None: #If we have a csv file as outputfile we write on it, else just print the result on the console
            #print("Writing on the csv File...")
            if pred_list[0][0]>0.5:
                theResultClass= 1
            else:
                theResultClass=0
            print(theResultClass)
            write_CSV_file(resultsFilePath,modeCSV, model_name, len(pred_list[0]), imgsPath,  pred_list,theResultClass)
        else:
            print()
            print(pred_list)
            print(">>THE PREDICTION FOR YOUR IMAGE <",imgsPath,"> IS:",np.argmax(pred_list))
            
            
    else: #ELSE if it's a directory of imgs
        myList = os.listdir(imgsPath)
        #print(myList
        for f in myList:
            #print(f)
            path=imgsPath+"/"+f
            pred_list = classifyImg_general(loaded_model,path) #Getting the result np array of predictions values, we then get the argmax
            
            if resultsFilePath!=None: #If we have a csv file as outputfile we write on it, else just print the result on the console
                #print("Writing on the csv File...")
                if pred_list[0][0]>0.5:
                    theResultClass= 1
                else:
                    theResultClass=0
                print(theResultClass)
                
                write_CSV_file(resultsFilePath,modeCSV, model_name, len(pred_list[0]), path,  pred_list,theResultClass)
            else:
                print()
                score=pred_list[0]
                print(score)
                print(
                    "This image is %.2f percent landscape and %.2f percent face."
                    % (100 * (1 - score), 100 * score)
                )
                    

def classifyImg_general(loaded_model,imgsPath):
    image_size = (150, 150)
    img = keras.preprocessing.image.load_img(imgsPath, target_size=image_size)
    plt.imshow(img)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    
    predictions = loaded_model.predict(img_array)
    print(predictions)
    return predictions
        
    
    
    

###############################################################################
#MNIST LOAD MODEL:
def loadModelMnist(modelName , imgToClassify = None,resultsFile=None,modeCSV=None):
    """
    Parameters
    ----------
    modelName : TYPE string
        DESCRIPTION. it's the models path
    imgToClassify : TYPE, optional :string
        DESCRIPTION. The default is None.
        It's the path of the image to be classify'

    Returns
    -------
    None.
    Prints in the console the classifier accuracy for test data 
    And classifies the imgToClassify if it was given to the argparser.
    If resultsFile is given, write the results on the csv file instead
    """
    
    num_classes=10
    mnist = tf.keras.datasets.mnist

    #Split data into training sets & test sets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    
    
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    
    
    ###LOAD THE MODEL:
    loaded_model= keras.models.load_model(modelName)

    
    #loaded_model.summary()
    
    loss, acc= loaded_model.evaluate(test_images,test_labels,verbose=0)
    print()
    print(">>The accuracy of the loaded model (testset) is: ",acc)

    if imgToClassify != None:
        print("We're going to classify your image...")
        #preprocessImg_Classify(loaded_model,imgToClassify)
        check_resultsMNIST(modelName,loaded_model,imgToClassify, resultsFile,modeCSV)
        print("Job finished.")
        
        
###############################################################################
#Utilities fcts;
    
    

def check_resultsMNIST(model_name, loaded_model,imgsPath,resultsFilePath,modeCSV):
    if(not(os.path.isdir(imgsPath))): #If the input is a simple file (a single img) 
        pred_list = preprocessImg_Classify(loaded_model,imgsPath)
        if resultsFilePath!=None: #If we have a csv file as outputfile we write on it, else just print the result on the console
            #print("Writing on the csv File...")
            theResultClass= np.argmax(pred_list[0])
            write_CSV_file(resultsFilePath,modeCSV, model_name, len(pred_list[0]), imgsPath,  pred_list, theResultClass)
        else:
            print()
            print(pred_list)
            print(">>THE PREDICTION FOR YOUR IMAGE <",imgsPath,"> IS:",np.argmax(pred_list))
            
            
    else: #ELSE if it's a directory of imgs
        myList = os.listdir(imgsPath)
        #print(myList
        for f in myList:
            #print(f)
            path=imgsPath+"/"+f
            pred_list = preprocessImg_Classify(loaded_model,path) #Getting the result np array of predictions values, we then get the argmax
            theResultClass= np.argmax(pred_list[0])
            if resultsFilePath!=None: #If we have a csv file as outputfile we write on it, else just print the result on the console
                #print("Writing on the csv File...")
                write_CSV_file(resultsFilePath,modeCSV, model_name, len(pred_list[0]), path,  pred_list,theResultClass)
            else:
                print()
                print(pred_list)
                print(">>THE PREDICTION FOR YOUR IMAGE ",path," IS:",np.argmax(pred_list))
        
        
#########################################################################################################################################
def write_CSV_file(csv_FILE,modeCSV, model_used_Name , nb_classes , imgPath , list_predsVals ,resultClassPred ):
    
    f=open(csv_FILE, modeCSV,newline="") #PARAMETER modeCSV (in argparser):'a' for append mode / 'w' to  create and write new one
    
    newLineToAppend = [model_used_Name,nb_classes,imgPath, convertListToStr(list_predsVals[0]),max(list_predsVals[0]), resultClassPred]#np.argmax(list_predsVals)]
    writer = csv.writer(f)
    writer.writerow(newLineToAppend)
     
    f.close()

###############################################################################

def preprocessImg_Classify(model, currentImgPath):
    """
    
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    currentImgPath : TYPE
        DESCRIPTION.

    Returns
    -------
    prediction : TYPE
        DESCRIPTION.

    """
    #gray = cv2.imread("StockImg/img5.png",cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread(currentImgPath,cv2.IMREAD_GRAYSCALE)
    #resize img and invert it (black background)
    gray = cv2.resize(255-gray , (28,28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #save the processed img
    #cv2.imwrite("StockImg/pro_img9.png",gray)
    #cv2.imwrite(imgPath[:-4]+"_pro"+imgPath[-4:],gray)
    flatten=gray.flatten()/255.0
    prediction = model.predict(flatten.reshape(1, 28, 28, 1))
    return prediction

def convertListToStr(L):
    """
    param: List of predictions
    return a string format for the list
    """
    string="[ "
    for elem in range(len(L)):
        if elem==len(L)-1:
            string= string+str(L[elem])
        else:
            string=string+str(L[elem])+" // "
        
    string = string+" ]"
    #print("CONVER LIST >>>",string)
    return string