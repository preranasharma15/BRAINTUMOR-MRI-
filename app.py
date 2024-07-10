import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import streamlit as st


st.title('Brain Tumor(MRI) Detection')
user_input = st.file_uploader("Click here to upload")
# st.image(user_input)

if user_input:
    img = np.array(user_input)


    model = keras.models.load_model('braintumor.h5')

   

    if isinstance(user_input, str):
        img = cv2.imread(user_input)
        if img is None:
            print("Error: Image not loaded.")
        else:
            resized_img = cv2.resize(img,(150,150))
            img_array = np.array(resized_img)
            img_array.shape

            img_array = img_array.reshape(1, 150, 150, 3)
            img_array.shape

            
            st.image(resized_img)
            # from PIL import Image
            # new_img = Image.open(resized_img)
            # new_img.show()

            a = model.predict(img_array)
            indices = a.argmax()
            st.write(a)
