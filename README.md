![](UTA-DataScience-Logo.png)

# Image Classification using Various CNNs
![image](https://user-images.githubusercontent.com/112579358/236504959-21f04375-28fe-43c8-8fdf-4a2b1abc96d6.png)

### Authors: Justin Pham, John Aguinaga, Jason Bard, Bavithra Lakshmanasamy

This repository holds an attempt to use multiple CNNs on multiple image datasets in order to successfully classify them.

## Overview

Our group tested four separate CNN architectures on three separate datasets. This includes a rock-paper-scissors (RPS) dataset, a natural scenery (Nature/Scenery) dataset, and a Driver Drowsiness Detection (Eyes) dataset, all freely available on Kaggle. Our architectures include one based on VGGNet, one based one AlexNet, one based on Dense121, and a custom architecture.

## Summary of Work Done

### Data

* Scenery Dataset:
  * Type: Image Data
    * Input: Scenery images (150x150 pixel jpges) 6 types of scenery: "Buildings", "Forest", "Glacier","Mountain","Sea" and "Street"
    * Input: Training/Testing Images, output: Training/Testing Labels.
  * Size: 399 MB total.
  * Instances: (Train, Test, Prediction split): Train: 14,000 images, Test: 3000 images, Prediction: 7300 images.
    * Prediction images did not have labels.
  * Available on [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
  
* Rock-Paper-Scissors (RPS) Dataset:
  * 2188 images of individual hand signals against a green background
    * 726 Rocks, 710 Papers, 752 Scissors
  * Stored in PNG format (200 x 300)
  * Data size: 160 MB
    * Images are included twice, leading to a true size of 321 MB.
  * Available on [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).
  
* Eyes Dataset:
  * The dataset contains 2467 images of human faces that depict different eye states (open or closed) and yawning behavior (yawning or not yawning).
  * Input: Images with eyes category (open, closed, yawn, no_yawn).
  * Output: Training/Testing Lables
  * Stored in JPG format (224 x 224)
  * Data size: 169 MB
  * Available on [Kaggle](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new)

#### Preprocessing / Clean up

Image processing methods are stored in the [`datasetload`](notebooks/datasetload.py) module.

* Scenery Dataset:
  * For the scenery dataset, the images were all of size 150x150 so in order to use with AlexNet architecture they had to be resized to 227x227 for proper usage. 
  * The libraries predominantly used for resizing were `Pandas` and `os`. Used file paths to iterate through folders in Google Drive, resized the image   using `os` and uploaded the images into lists. 
  * After shuffling, normalizing and one-hot encoding, the lists were converted into arrays for the Machine Learning algorithm.
  
* RPS Dataset:
  * Images did not have to be resized since they were all 200x300. Instead, the neural network architecture shifted to allow for these to slot perfectly into the input layer.
  * The main library used for opening the images with `PIL`. The images were shuffled, normalized, and split into training and testing sets..
  
* Eyes Dataset:
  * As the images in the dataset were of varying sizes without any standard dimensions, they were resized to 224x224 to ensure compatibility with the Densenet model.
  * The data was pre-processed using libraries like `openCV` and `Pandas`. The pixel values were normalized and the labels were one-hot encoded. 
   
#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

* Scenery Dataset:  
  * Picture randomly chosen from training dataset after one-hot encoding:  
   ![RandomPicture](pictures/AlexNet/sceneimage.png)
   
* RPS Dataset:

From left to right: Rock, Paper, Scissors.

![rock](pictures/VGGNet/rock.png) ![paper](pictures/VGGNet/paper.png) ![scissors](pictures/VGGNet/scissor.png)

* Eyes Dataset:
  * Loaded pictures from Training dataset
  ![Open_closed](https://user-images.githubusercontent.com/112579358/236497468-a4a29adf-b8ae-4e36-b587-8c99c223565d.png)
  ![yawn_noyawn](https://user-images.githubusercontent.com/112579358/236497713-9ccaae3c-eea3-4002-9d5b-7c3c9f754471.png)


     
### Problem Formulation

#### Datasets

* Scenery Dataset:
  * Input: Images of scenery 227x227 pixels (jpg).
  * Output: Classification one-hot label
  
* RPS Dataset:
  * Input: Images of rock-paper-scissors 200x300 pixels (png)
  * Output: Classification one-hot label
  
* Eyes Dataset:
  * Input: Images of Eyes 224x224 pixels (jpg)
  * Output: Classification one- hot label

Model architectures are also stored in the `datasetload` module, although they were copied and tweaked in the individual notebooks as well.

#### Models

* AlexNet: A convolutional neural network containing eight layers; the first five are convolutional layers, some of them followed by max-pooling layers, and the last three were fully connected layers. 
   * Loss: Categorical_Crossentropy
   * Optimizer: Adam
   * Other hyperparameters: Learning rate = 0.001, Dropout = 0.5
   * Training: The model was created using Keras and was made Sequentially. Then it was instantiated and trained via Google Colaboratory. The training took 5    minutes per dataset, with 30 epochs and a batch size of 64.
     
* VGGNet: Contains five blocks of convolutions separated by max-pooling, followed by a seven-layered DNN.
  * Loss: Categorical Cross-entropy
  * Optimizer: Adam
  * Hyperparameters: Learning rate = 0.001
  * Training: The model was created Sequentially through keras. It was training in Google Colab. Data loading and training took 15 minutes for each model, with 20 epochs.
  
* DenseNET121: Convolutions seperated by Average pooling, followed by four dense blocks where inputs are passed on to and concatenated to pass on to the next block
  * Loss: Categorical Cross-entropy
  * Optimizer: Adam
  * Hyperparameters: Learning rate = 0.001
  * Training: The model was constructed using Keras' Functional API and trained on Google Colab. Each model took approximately 15-20 minutes to train, with 30 epochs being used.
  
* Custom Arch: Contains two blocks of convolutions seperated by max-pooling, followed by two-layered DNN. 
  * Loss: Categorical Cross-entropy
  * Optimizer: Adam
  * Hyperparameters: Learning rate = 0.001, Dropout = 0.4
  * Training: The model was created Sequentially through keras. It was training in Google Colab. Data loading and training took 4 minutes for each model, with 10 epochs and a batch size of 16. 
 
### Performance Comparison

#### AlexNet

![results](pictures/AlexNet/AlexNet_measures.png)    
    
<img src="pictures/AlexNet/Scenery dataset visuals/loss_curve_and_accuracy (scenery).png" width="300" height="300"/>   <img src="pictures/AlexNet/RPS dataset visuals/loss_curve_and_accuracy (RPS).png" width="300" height="300"/>   <img src="pictures/AlexNet/Eyes dataset visuals/loss_curve_and_accuracy (eyes).png" width="300" height="300"/>
     

#### VGGNet

| Dataset          |  Accuracy  |   Precision |   Recall |       F1 |
|:----------------:|:----------:|:-----------:|:--------:|:--------:|
| RPS              |   0.337386 |    0.113829 | 0.337386 | 0.170227 |
| Nature / Scenery |   0.175000 |    0.030625 | 0.175000 | 0.052128 |
| Eyes             |   0.251732 |    0.063369 | 0.251732 | 0.10125  |

![loss-rps](pictures/VGGNet/loss-rps.png) ![loss-nature](pictures/VGGNet/loss-nature.png) ![loss-eyes](pictures/VGGNet/loss-eyes.png)

#### DenseNet121

![results git](https://user-images.githubusercontent.com/112579358/236503459-75d924b7-62c0-4136-8bcf-1e082e18eef9.png)

![curves git](https://user-images.githubusercontent.com/112579358/236503745-e7472720-df5a-437d-8650-c9dbd72b5de4.png)



#### Custom Arch

![results](pictures/CustomArch/results_image1.png)

![loss-rps](pictures/CustomArch/loss_rps.png) ![loss-nature](pictures/CustomArch/loss_nature.png) ![loss-eyes](pictures/CustomArch/loss_eyes.png)

### Conclusions
* Unsure about what happened with VGGNet.
 * All other datasets had accuracy that was substantially better than random guessing

* All other models are effective especially on the RPS dataset. The pixel values are more uniform. 

* Densenet works better for RPS with 0.998 compared to Eyes and Nature.
* Alexnet and Dense121 worked best for Nature with ~0.76 accuracy. 
* Custom Arch worked best for eyes with ~0.84 accuracy.


### Future Work

* Add a confusion matrix to evalutate and analyze the models performance.
* Lower the parameters for each model to increase performance and accuracy. 
* Train more images with higher number of epochs
* Split a small part of the training set for validation data. 

## How to reproduce results

* Google Colab Pro may be needed, which costs $9.99 a month. 

### Overview of files in repository

* Folder: Notebooks
 * Contains all models used as well as module. 
* Folder: Pictures
 * Contains pictures and graphs used in readme. 

### Software Setup
* `keras, sklearn, pandas, matplotlib, tensorflow, cv2, os, numpy, glob, PIL, opendatasets`
* To install packages: `!pip install opendatasets` and import the rest. 

## Citations

* https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98  
* https://medium.com/analytics-vidhya/vggnet-architecture-explained-e5c7318aa5b6
* https://www.kaggle.com/code/blurredmachine/alexnet-architecture-a-complete-guide/notebook
* https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c 
* https://www.kaggle.com/code/dansbecker/deep-learning-from-scratch-daily 
* https://arxiv.org/abs/1608.06993
* https://keras.io/api/applications/densenet/
* https://iq.opengenus.org/architecture-of-densenet121/
* https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803








