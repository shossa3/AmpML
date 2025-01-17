# Diagnostic Categorization using Mel Frequency Cepstral Coefficients and Convolutional Neural Networks

# Overview

This project uses the Respiratory Sound Database for diagnostic categorization by means of a Convolutional Neural Network (CNN) model architecture implemented in Keras using a Tensorflow backend and Mel Frequency Cepstral Coefficients (MFCCs) as the extracted features. After doing some preliminary exploratory data analysis on the signal durations and frequencies, I chose MFCCs as my extracted feature of choice given that they are a compact, robust, and computationally efficient spectral representation of audio. Furthermore, the choice of CNNs is an intuitive choice for dealing with 2D spectrograms data. 

I designed a simple sequential model, consisting of four Conv2D convolution layers, with the final output layer being a dense layer. The convolution layers are designed for feature detection. They work by convolution - where a sliding filter window moves over the input and performs a matrix multiplication and stores the result in a feature map. The filter parameter specifies the number of nodes in each layer. Each layer increases in size from 16, 32, 64 to 128, while the kernel_size parameter specifies the size of the kernel window which in this case is 2 resulting in a 2x2 filter matrix.

The first layer receives the input shape of (40, 862, 1) where 40 is the number of MFCC's, 862 is the number of frames (taking padding into account) and the 1 due to the audio being mono. The activation function used for the convolutional layers is ReLU. A small Dropout value of 20% was used on the convolutional layers. Each convolutional layer has an associated pooling layer of MaxPooling2D type with the final convolutional layer having a GlobalAveragePooling2D type. The pooling layer is to reduce the dimensionality of the model (by reducing the parameters and subsquent computation requirements) which serves to shorten the training time and reduce overfitting. The Max Pooling type takes the maximum size for each window and the Global Average Pooling type takes the average which is suitable for feeding into our dense output layer.

The output layer will have 6 nodes (num_labels) which matches the number of possible classifications. The activation is for the output layer is softmax. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.

Spectogram of MFCCs: 
![Figure_1](https://github.com/user-attachments/assets/1ae3485a-fa1e-428c-940d-21824bbf3250)

# Training 

The data was split into training and testing sets using the train_test_split function from Scikit. The adam optimizer is used to adapt the learning rate of the gradient descent process. The metric chosen was accuracy. The loss function I chose was categorical crossentropy which is commonly used in deep learning models for multi-class classification problems. It measures the difference between the model's predicted probabilities and the true labels. The distribution count of occurence of the different diseases shows that COPD is the most frequent diagnosis. 

Distribution of classes:
![Figure_2](https://github.com/user-attachments/assets/16a8b7a5-758a-40f2-825a-2012f588c049)


ROC Curves for each class - True Positive Rate vs False Positive Rate 
![Figure_3](https://github.com/user-attachments/assets/a648d0e0-64c5-4c96-ae78-edf5adcc0c4d)

I used the sklearn ROC extension to multiclass for this dataset. See below for the classification metrics.  <br /> 

Classification metrics:  <br /> <img width="526" alt="Screenshot 2025-01-15 at 12 43 58 PM" src="https://github.com/user-attachments/assets/a291255e-8bdb-4288-ae32-0028baf6a7df" /> 


# Scripts 

Classification.py - contains the model implementation, data loading, and traning.  <br /> Inference.py - consists of an example of loading the model and running inference on example audio.  <br /> 

# Setup 

This project uses Python 3 and requires the installtion of the following packages: numpy, librosa, keras, tensorflow, pandas, sklearn, matplotlib, and seaborn. Place the wav and text files from the Respitory Sound Database in a folder named "audio". First run the classification script which will generate the HDF5 file model files after training is completed. Once training is completed, you can specify the model name to load in the inference script and also replace the example audio file with one of your choosing.   
