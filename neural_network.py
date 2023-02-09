'''
Developed by Tomas Medeot, February 2023
Email: tomimedeot@gmail.com
'''

import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image


def image_load(path:str,shuffle:bool)->tuple[np.ndarray,np.ndarray,list[str]]:
    '''
    Load images from the folders on directory from -path-
    The labels are grouped accordig to images array
    You can shuffle using -True- in -shuffle- to reorder the images and labels equally
    '''
    images = []
    labels = []
    label = 0
    print("Loading images")
    categories = os.listdir(path)
    print("Categories founded:")
    for counter_i in categories:
        print("   "+counter_i)
    for type in categories:
        for img in os.listdir(f"{path}{type}"):
            image = Image.open(f"{path}{type}/{img}").resize((100,100))
            image = image.convert("L")
            image = tf.cast(image, tf.float32)
            image /= 255
            image = np.asarray(image)
            images.append(image)
            labels.append(label)
        label += 1
    images = np.asarray(images, dtype="float32")
    labels = np.asarray(labels, dtype="float32")
    if shuffle:
        new_order=[]
        new_images=[]
        new_labels=[]
        for counter_i in range(len(images)):
            new_order.append(counter_i)
        random.shuffle(new_order)
        for counter_e in new_order:
            new_images.append(images[counter_e])
            new_labels.append(labels[counter_e])
        images = np.asarray(new_images, dtype="float32")
        labels = np.asarray(new_labels, dtype="float32")

    return images,labels,categories


def neural_network()->object:
    '''
    Set the neural network with 10.000 inputs and 2 outputs
    The activaction functions are ReLu for the input and hiden layers, for output layer is SoftMax
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3), input_shape=(100,100, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),

        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train(model:object,images:np.ndarray,labels:np.ndarray,model_name:str)->tuple[object,object]:
    '''
    Train the Neural Network Model using the images and labels and then save
    '''
    print("Training")
    trained_model = model.fit(images, labels, epochs=28)
    model.save(model_name)
    return model,trained_model


def test(model:object,images:np.ndarray,labels:np.ndarray,categories:list)->str:
    '''
    Test the accuracy from Neural Network Model previusly trained and count the mistakes
    '''
    errors = 0
    prediction = np.argmax(model.predict(images), axis=1)
    for counter_i in range(len(prediction)):
        if prediction[counter_i] != labels[counter_i]:
            errors += 1
        print(categories[prediction[counter_i]])
    return str(errors)+' Mistakes'


def predict(model:object,categories:list,path:str,img:str)->str:
    '''
    Select a photo from directory -path- and name -img- to predict in categories
    '''
    images = []
    image = Image.open(f"{path}/{img}").resize((100,100))
    image = image.convert("L")
    image = tf.cast(image, tf.float32)
    image /= 255
    image = np.asarray(image)
    images.append(image)
    images = np.asarray(images, dtype="float32")
    prediction = np.argmax(model.predict(images))
    return categories[prediction]


if __name__ == '__main__':

    model =neural_network()#Set model

    images, labels, categories =image_load("DataSet/Train/", True)#Load images to train

    model, trained =train(model,images,labels,'Trained_Model')#Train-save the model for predictions

    test_images, test_labels, test_categories =image_load("DataSet/Test/",True)#Load test images

    print(test(model,test_images,test_labels,test_categories))#Try efectivty
