#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the functions to train the models called by "mainBolsa.py"

Refs:
    https://keras.io/examples/vision/image_classification_from_scratch/
    https://keras.io/guides/transfer_learning/
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow import keras
import cv2


def train_transfer_learning(saveFileName, inputPATH_DATA, epochs):
    """
    Trains a model based on the transfer learning technique and fine tuning;
    ->binary class for now;
    
    Parameters
    ----------
    saveFileName : TYPE string
        DESCRIPTION. name of the model to be saved
    inputPATH_DATA : TYPE 
        DESCRIPTION.
    """
    
    image_size = (150, 150)
    batch_size = 32
    
    #Generate the datasets, there are some requirements for the function that follows in terms of hierarchy:
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        inputPATH_DATA,#"FaceOrNoFace",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        inputPATH_DATA,#"FaceOrNoFace",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    #map a function of resizing
    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, image_size), y))
    val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, image_size), y))
    
    
    train_ds = train_ds.prefetch(buffer_size=32) #useful for systems pipeline based
    val_ds = val_ds.prefetch(buffer_size=32)#it starts fetching the next data while classifying the current one


    #Introducing "sample diversity" by applying random transformations. 
    #Improve classifier by showing new aspects and by reducing overfitting.
    data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ]
    )
    
    #TRANSFER LEARNING WORKFLOW:
        #1) instantiate a base model and load pre-trained weights into it;
        #2) Freeze all layers in the base model by setting;
        #3) Create a new model on top of the output of one (or several) layers from the base model.
        #4) train your new model on your new dataset.
    
    base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
    )  # Do not include the ImageNet classifier at the top.
    
    # Freeze the base_model
    base_model.trainable = False
            
    #Creation of new model ;
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation
    
    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.), the normalization layer
    # does the following, outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([127.5] * 3)
    var = mean ** 2
    # Scale inputs to [-1, +1]
    x = norm_layer(x)
    norm_layer.set_weights([mean, var])
    
    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    

    model.summary()

    #BatchNormalization contains 2 non-trainable weights that get updated during training.
    #These are the variables tracking the mean and variance of the inputs.

    
    #Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    #Training the new top layer
    #epochs = 5
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    #NEXT STEP(optional in practice)
    #FINE TUNING: unfreeze the model and train it entirely with a low learning rate;
    
    #running in inference mode => batch norm layers dont update their batch statistics
    
    base_model.trainable = True
    model.summary()

    model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    )
    
    
    nbepochs = 3
    model.fit(train_ds, epochs=nbepochs, validation_data=val_ds)

    print()
    if saveFileName!=None:
        model.save(saveFileName+'.h5')
        print("TRAINING OF THE MODEL & SAVING PROCESS IS OVER...")




###############################################################################

### MNIST FUNCTIONS


def train_default_mnist(saveFileName ):
    """
    Parameters
    ----------
    saveFileName : TYPE string
        DESCRIPTION. name of the model to be saved.
    Returns
    -------
    None.
    SAVES THE MODEL IN THE PATH GIVEN BY saveFileName
    """
    
    
    print("BEGINNING TRAINING...")
    mnist = keras.datasets.mnist
    num_classes=10
    input_shape = (28, 28, 1)
    #Split data into training sets & test sets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    #Scale all values from [0;255] to [0;1] 
    
    #train_images = train_images / 255.0
    #test_images = test_images / 255.0
    # Scale images to the [0, 1] range
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
        
        
    # convert class vectors to binary class matrices
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels =  keras.utils.to_categorical(test_labels, num_classes)



    
    # "Sequential layers dynamically adjust the shape of input to a layer based the out of the layer before it"
    model = keras.Sequential([
    keras.Input(shape=input_shape),
    keras.layers.Flatten(),  # input layer (1)input_shape=(28, 28)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(num_classes, activation='softmax') # output layer (3)
    ])
    
    
    #COMPILE MODEL
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',#'sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    #TRAINING:
    #fit the model to the training data
    model.fit(train_images, train_labels, epochs=10)
    
    #Test model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    print("The accuracy of the classifier default is:",test_acc)
    
    #Saving the model .h5
    #tmpString = 'Pretrained_models'+saveFileName+'.h5'
    #model.save('Pretrained_models/'+saveFileName+'.h5')
    print()
    if saveFileName!=None:
        model.save(saveFileName+'.h5')
        print("TRAINING OF THE DEFAULT MODEL & SAVING PROCESS IS OVER...")
    
    else:
        print("TRAINING OF THE DEFAULT MODEL IS OVER...")


###############################################################################
#CNN from keras;
def train_CNN_mnist(saveFileName ):
    """
    Parameters
    ----------
    saveFileName : TYPE string
        DESCRIPTION. name of the model to be saved.
    imageToTest : TYPE, optional
        DESCRIPTION. The default is None.
        name of the image to test after the training phase (path)
    Returns
    -------
    None.
    SAVES THE MODEL IN THE PATH GIVEN BY saveFileName

    """
    #Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    
    mnist = keras.datasets.mnist
    #Split data into training sets & test sets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Scale images to the [0, 1] range
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
        
        
    # convert class vectors to binary class matrices
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels =  keras.utils.to_categorical(test_labels, num_classes)
    
    
    ##############
    # BUILD MODEL
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
    )
    ##############
    #Compile & training
    batch_size = 128
    epochs = 15
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    #Eval
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    
    
    
    print("The accuracy of the classifier CNN is:",test_acc)
    
    
    
    
    #Saving the model .h5
    #tmpString = 'Pretrained_models'+saveFileName+'.h5'
    #model.save('Pretrained_models/'+saveFileName+'.h5')
    print()
    if saveFileName!=None:
        model.save(saveFileName+'.h5')
        print("TRAINING OF THE CNN & SAVING PROCESS IS OVER...")
    else:
        
        print("TRAINING OF THE CNN IS OVER...")


    
    
    
"""    
    
def preprocessImg_Classify(model, imgPath):
    #gray = cv2.imread("StockImg/img5.png",cv2.IMREAD_GRAYSCALE)
    
    gray = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    #resize img and invert it (black background)
    gray = cv2.resize(255-gray , (28,28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #save the processed img
    #cv2.imwrite("StockImg/pro_img9.png",gray)
    cv2.imwrite(imgPath+"_pro",gray)
    


    flatten=gray.flatten()/255.0
    prediction = model.predict(flatten.reshape(1, 28, 28, 1))
    print()
    print(prediction)
    print(">>THE PREDICTION FOR YOUR IMAGE IS:",np.argmax(prediction))
    
"""
    
    