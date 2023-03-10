# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nWHRno9T0wv5smwUVQV-KH4bk7FSdmMT
"""

import os
import pandas as pd
import numpy as np
from matplotlib import image

import seaborn as sns
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.io import imread
from skimage.color import rgb2gray

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from google.colab import drive
drive.mount('/content/drive')

pathtest = '/content/drive/MyDrive/Alzheimer_s Dataset-2/test'
pathtrain = '/content/drive/MyDrive/Alzheimer_s Dataset-2/train'

class1 = '/content/drive/MyDrive/Alzheimer_s Dataset-2/train/VeryMildDemented' 
class2 = '/content/drive/MyDrive/Alzheimer_s Dataset-2/train/MildDemented'
class3 = '/content/drive/MyDrive/Alzheimer_s Dataset-2/train/ModerateDemented'
class0 = '/content/drive/MyDrive/Alzheimer_s Dataset-2/train/NonDemented'

def file_append(class_path):    
  image_array = []    
  curr_path = os.path.join(os.getcwd(),class_path)  
  cnt = 0
  file_list = [k for k in os.listdir(curr_path) if '.jpg' in k]

def file_append(class_path):
              img_path = os.path.join
              curr_path:any
              img = imread(img_path)
              img = resize(img,(60,60,3))
              img = img.flatten()
              image_array:any

"return" 
image_array:any
class1_img = str(class1)
class2_img = str(class2)
class3_img = str(class3)
class0_img = str(class0)

from ast import Str
int:str
class_path:any
int:str = 1

from traitlets.traitlets import Instance
from typing import Any
from tables.table import default_index_filters
default_index_str:Any
int
def_init:any;
int
df=2;
df3 =default_index_filters
filters:df3
default_index_filters : any
df0:any

shape:any

plot:Any

from pandas.core.arrays.interval import value_counts
value_counts
plt.ylabel('Count')
plt.title('Class Wise Distribution of data')