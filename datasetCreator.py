#import libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

#defining basic variables 
baseDirectory = "./Dataset/"
imageSize = 220 #the default NN1 input size discussed in FaceNet paper
data = []

#generating dataframe from images
for name in os.listdir(baseDirectory):
    for image in os.listdir(baseDirectory+name):
        image = cv2.imread(os.path.join(baseDirectory+name,image))
        image = cv2.resize(image, (imageSize,imageSize))
        data.append((image, name))

#function to create dataframe and return it
def dataframeCreator():
    dataframe = pd.DataFrame(data, columns=["image","name"])
    return dataframe 