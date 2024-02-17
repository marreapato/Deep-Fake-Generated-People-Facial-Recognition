import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

#base = 'D:/gitrepos/Deep-Fake-Generated-People-Facial-Recognition/streamlit-app'
model = keras.models.load_model('model_dfake-face_softmax.h5')

def image_pre(image_data):
    img = img = cv2.imdecode(np.frombuffer(image_data, np.float32), -1)
    img = cv2.resize(img,(128,128))
    img = np.array(img)
    data = img.reshape((-1,128,128,3))
    return data

def predict(data):
    prediction = model.predict(data)
    return prediction[0][1]
