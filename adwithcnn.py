# -*- coding: utf-8 -*-
"""ADwithCNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SR0J5IU9R8ktkWMnrkY49MQXra7PBzaR
"""

from google.colab import drive

drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import random 
import cv2
import keras
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras.activations import sigmoid
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
from keras.models import Sequential

#train path
train_p = "/content/drive/MyDrive/Alzheimer_s Dataset/train"
#test path
test_p = "/content/drive/MyDrive/Alzheimer_s Dataset/test"

train_datagen = ImageDataGenerator(validation_split=0.1,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_batches = train_datagen.flow_from_directory(directory=train_p, 
                                                  classes=['NonDemented', 'VeryMildDemented', 
                                                           'MildDemented', 'ModerateDemented'], 
                                                  target_size=(224, 224),
                                                  subset='training', 
                                                  batch_size=10)

validation_batches = train_datagen.flow_from_directory(directory=train_p, 
                                                       classes=['NonDemented', 'VeryMildDemented', 
                                                                'MildDemented', 'ModerateDemented'], 
                                                       target_size=(224, 224),
                                                       subset='validation',
                                                       batch_size=10)

test_datagen = ImageDataGenerator(rescale=1./255)

test_batches = test_datagen.flow_from_directory(directory=test_p, 
                                                classes=['NonDemented', 'VeryMildDemented', 
                                                         'MildDemented', 'ModerateDemented'], 
                                                target_size=(224, 224),
                                                batch_size=10, 
                                                shuffle=False)

"""I added data augmentation to the training data generator, which includes rescaling, shearing, zooming, and horizontal flipping of the images. I also specified the target_size parameter to resize the images to a common size of 224 x 224 pixels, which is a commonly used size for image classification models.

I also separated the training data generator and the test data generator to have different parameters. For the test data generator, I only included rescaling to normalize the pixel values.

Overall, these changes should improve the performance of the machine learning model by augmenting the training data and resizing the images to a common size.
"""

class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
num_images = []

for cls in class_names:
    path = os.path.join(train_p, cls)
    num_images.append(len(os.listdir(path)))

fig, ax = plt.subplots()
ax.bar(class_names, num_images, color=(0.7, 0.2, 0.4, 0.9))
ax.set_title('Number of Images per Class')
ax.set_xlabel('Class')
ax.set_ylabel('Number of Images')

"""I renamed the data dictionary to num_images to better reflect the information it stores. I also used os.path.join() to construct the path to the image directories, which is more platform-independent.

Instead of using two loops to iterate through each class and each image in the training data, I simply looped through the class_names list and used os.listdir() to count the number of images in each class directory. The resulting counts are stored in the num_images list.

Finally, I used the subplots() function to create a figure with a single subplot, and passed the ax object to the bar() function to plot the data. I also added a title, and x and y axis labels to the plot for clarity.
"""

img_size = 224
num_classes = 4
model = Sequential([
    layers.Input((img_size, img_size, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(300, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(num_classes, activation='softmax')
])

metrics = [keras.metrics.CategoricalAccuracy(name='accuracy'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.AUC(name='auc')]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=metrics)

epo = 5
b_size = 8

history = model.fit(x=train_batches,
                    validation_data=validation_batches,
                    steps_per_epoch=len(train_batches),
                    validation_steps=len(validation_batches),
                    epochs=epo,
                    batch_size=b_size, 
                    verbose=2)

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()
            
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
    def compute_heatmap(self, image, eps=1e-8):

        gradModel = tf.keras.Model(
        inputs=[self.model.inputs],
        outputs=[self.model.get_layer(self.layerName).output,
                 self.model.output])
      
        with tf.GradientTape() as tape:

            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        
      
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
    
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):

        heatmap = cv2.applyColorMap(heatmap, colormap)
        image = np.asarray(image, np.float64)
        heatmap = np.asarray(heatmap, np.float64)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)

def show_heatmap(model, img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    i = np.argmax(preds[0])
    label_to_class = {'NonDemented': 0,
                      'VeryMildDemented': 1,
                      'MildDemented': 2,
                      'ModerateDemented': 3}

    class_to_label = {v: k for k, v in label_to_class.items()}

    label = class_to_label[i]
    print(f'Predicted class: {label} | Prediction probability: {max(preds[0]) * 100}%')
    
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)

    (heatmap, output) = cam.overlay_heatmap(heatmap, image[0], alpha=0.5)
    
    output = output.astype(np.uint8)
    plt.imshow(output, interpolation='nearest')
    plt.show()

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import img_to_array

img_path = "/content/drive/MyDrive/Alzheimer_s Dataset/test/NonDemented/26 (64).jpg"
show_heatmap(model,img_path)

img_path = "/content/drive/MyDrive/Alzheimer_s Dataset/test/MildDemented/26 (19).jpg"
show_heatmap(model,img_path)