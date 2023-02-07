import os

import numpy as np
import tensorflow as tf
from PIL import Image
import random


def Image_Load(path: str, shuffle:bool):
    '''
    Load images from the folders on directory from -path-
    The labels are grouped accordig to images array
    You can shuffle using -True- in -shuffle- to reorder the images and labels equally
    '''

    images = []
    labels = []
    label = 0
    
    print("\nLoading images")
    categories = os.listdir(path)
    print("Categories founded:")
    for i in categories:
        print("   "+i)

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
        for i in range(len(images)):
            new_order.append(i)
        random.shuffle(new_order)#,random.random
        for e in new_order:
            new_images.append(images[e])
            new_labels.append(labels[e])
        images = np.asarray(new_images, dtype="float32")
        labels = np.asarray(new_labels, dtype="float32")

    return images, labels, categories

def Neural_Network():
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

def Train(model: object, images: np.ndarray, labels: np.ndarray, model_name: str):
    '''
    Train the Neural Network Model using the images and labels and then save
    '''
    print("Training")
    trained_model = model.fit(images, labels, epochs=28)
    model.save(model_name)

    return model, trained_model

def Test(model: object, images: np.ndarray, labels: np.ndarray, categories: list):
    '''
    Test the accuracy from Neural Network Model previusly trained and count the mistakes
    '''
    errors = 0

    prediction = np.argmax(model.predict(images), axis=1)
    for i in range(len(prediction)):
        if prediction[i] != labels[i]:
            errors += 1
        print(categories[prediction[i]])
    print(str(errors)+' Mistakes')

def Predict(model, categories: list, path: str, img: str):
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

    model = Neural_Network() #Set model

    images, labels, categories = Image_Load("DataSet/Train/", True) #Load the images to train

    model, trained = Train(model, images, labels, 'Trained_Model') #Train the model for predictions

    test_images, test_labels, test_categories = Image_Load("DataSet/Test/", True) #Load the test images

    Test(model, test_images, test_labels, test_categories) #Try efectivty

    model.save('First test') #Save the trained model to use to use at any time