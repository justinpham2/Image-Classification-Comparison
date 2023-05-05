import opendatasets as od
import PIL
import glob
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Activation, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

def mount_drive(fr = False):
    from google.colab import drive
    drive.mount("/content/drive/", force_remount = fr)
    
def fetch_rps(shuffle = False, train_test_split = True, random_state = 0, train_size = 0.8):

    mount_drive(False)
    
    rocklst, paperlst, scissorslst = glob.glob("drive/My Drive/DATA4380-project/rockpaperscissors/rock/*"), glob.glob("drive/My Drive/DATA4380-project/rockpaperscissors/paper/*"), glob.glob("drive/My Drive/DATA4380-project/rockpaperscissors/scissors/*")
    
    def form_array(arr):

        a = []
        arr = np.asarray(arr)
           
        for i in arr:
            zq = np.asarray(PIL.Image.open(i))
            a.append(zq)
        

        return np.array(a)/255.0

        ## Turning each list into a 4D array (n, 200, 300, 3) 
        ## which contains the RGB images as arrays.
        ## We can take each 3D array contained within the larger structure as an individual input.
        ## Our targets will be categorical based on the list that it came out of.

    rock, paper, scissors = form_array(rocklst), form_array(paperlst), form_array(scissorslst)
    
    labels = pd.get_dummies(pd.Series(np.array([1]*len(rock) + [2]*len(paper) + [3]*len(scissors)))).to_numpy() 
    #one-hot encoding all of the labels made up from a list
    
    images = np.concatenate((rock,paper,scissors), axis = 0)
    
    if(shuffle):
        images, labels = sklearn.utils.shuffle(images, labels, random_state = random_state)
    
    if(train_test_split):
        image_train, image_test, label_train, label_test = sklearn.model_selection.train_test_split(images, labels, random_state = random_state, train_size = train_size, shuffle = True)
        
        return image_train, label_train, image_test, label_test
    
    else:
        return images, labels
    
def fetch_nature():
    
    mount_drive(False)
    
    # Choose your image size
    # AlexNet
    IMAGE_SIZE = (227, 227)
    
    output = []

    class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
    index  = [0, 1, 2, 3, 4, 5]

    # Build a dictionary that maps the category to an index/integer

    class_names_label = dict(list((zip(class_names, index))))

    # Iterate through training and test sets
    def function1(dataset):
      
        images = []
        labels = []

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in os.listdir(os.path.join(dataset, folder)):

                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # This function specifically is what resizes, IMAGE_SIZE is defined above
                image = cv2.resize(image, IMAGE_SIZE) 

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   

        # Shuffling
        images, labels = sklearn.utils.shuffle(images, labels)

        # Normalizing data
        images = images / 255.0
        
        #One-hot encoding
        labels = pd.get_dummies(pd.Series(labels)).to_numpy()

        return images, labels

    (image_train, label_train), (image_test, label_test) =  function1('/content/drive/MyDrive/DATA4380-project/Natural_Imagery_Dataset/seg_train/seg_train'), function1('/content/drive/MyDrive/DATA4380-project/Natural_Imagery_Dataset/seg_test/seg_test')

    return image_train, label_train, image_test, label_test
    
def fetch_eyes():

    mount_drive(False)
    
    # Choose your image size
    # AlexNet
    IMAGE_SIZE = (224, 224)
    
    output = []

    class_names = ['Open', 'Closed', 'no_yawn', 'yawn']
    index  = [0, 1, 2, 3]

    # Build a dictionary that maps the category to an index/integer

    class_names_label = dict(list((zip(class_names, index))))

    # Iterate through training and test sets
    def function1(dataset):
      
        images = []
        labels = []

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in os.listdir(os.path.join(dataset, folder)):

                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # This function specifically is what resizes, IMAGE_SIZE is defined above
                image = cv2.resize(image, IMAGE_SIZE) 

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   

        # Shuffling
        images, labels = sklearn.utils.shuffle(images, labels)

        # Normalizing data
        images = images / 255.0 

        # One- hot Encoding
        labels = pd.get_dummies(pd.Series(labels)).to_numpy()

        return images, labels

    (image_train, label_train), (image_test, label_test) =  function1('/content/drive/MyDrive/DATA4380-project/yawn-eye-dataset-new/dataset_new/train'), function1('/content/drive/MyDrive/DATA4380-project/yawn-eye-dataset-new/dataset_new/test')

    return image_train, label_train, image_test, label_test

def vgg_model(IMAGE_SIZE = (227,227,3)):

    #Based on model from https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

    model = Sequential([
        Conv2D(input_shape = IMAGE_SIZE, filters=16,kernel_size=(3,3),padding="same", activation="relu"),
        Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2),strides=(2,2)),
        Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2),strides=(2,2)),
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2),strides=(2,2)),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2),strides=(2,2)),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPool2D(pool_size=(2,2),strides=(2,2)),
        Flatten(),
        Dense(units=2048,activation="relu"),
        Dense(units=4096,activation="relu"),
        Dense(units=512, activation="relu"),
        Dense(units=256, activation="relu"),
        Dense(units=120, activation="relu"),
        Dense(units=24, activation="relu"),
        Dense(units=3, activation="softmax"),
        ])
        
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return model

def custom_model(IMAGE_SIZE = (227,227,3)):

  model = keras.models.Sequential([
    Conv2D(filters=128,kernel_size = 10, activation = 'relu',input_shape=IMAGE_SIZE),
    MaxPooling2D(pool_size = 3),
    Conv2D(filters=96,kernel_size = 5, activation = 'relu'),
    MaxPooling2D(pool_size = 3),
    Flatten(),
    Dense(900, activation='relu'),
    Dropout(0.4),
    Dense(500, activation='softmax'),
    Dropout(0.4),
  ])

  model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

  return model
  
def AlexNetModel(IMAGE_SIZE = (227,227,3)):    

  # Based on the model from: https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
  
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=95, kernel_size=(10,10), strides=(4,4), activation='relu', input_shape=IMAGE_SIZE),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=160, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=205, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=205, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=150, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return model

def dense121(IMAGE_SIZE = (227,227,3)):

    def conv_block(x, growth_rate):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(growth_rate, 3, padding='same')(x)
        return x

    def dense_block(x, layers, growth_rate):
        for i in range(layers):
            conv = conv_block(x, growth_rate)
            x = concatenate([x, conv])
        return x

    def transition_layer(x, filters):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, 1)(x)
        x = AveragePooling2D(2, strides=2)(x)
        return x
        
    # Define input layer
    inputs = tf.keras.Input(shape=IMAGE_SIZE)
    
    # Preprocessing layers
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # DenseBlock 1
    x = dense_block(x, 6, 32)
    # Transition Layer 1
    x = transition_layer(x, 128)
    
    # DenseBlock 2
    x = dense_block(x, 6, 32) # increase growth rate
    # Transition Layer 2
    x = transition_layer(x, 228)
    
    # DenseBlock 3
    x = dense_block(x, 12, 128) # increase growth rate
    # Transition Layer 3
    x = transition_layer(x, 478)
    
    # DenseBlock 4
    x = dense_block(x, 16, 128) # increase growth rate
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Build the model
    model = Model(inputs, outputs, name='densenet121')
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return model