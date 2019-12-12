# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:37:08 2019

@author: ASUS
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import model_from_json

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/WholeDamage.h5'
#MODEL_PATH = 'models/WholeDamage.h5'




# Load your trained model
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')





def loadDamageModel():
    json_file = open('Damage.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Damage.h5")
    
    #compile and evaluate loaded model
    
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = tf.compat.v1.get_default_graph()
    return loaded_model, graph


def loadFrontModel():
    json_file = open('front.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("front.h5")
    
    #compile and evaluate loaded model
    
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = tf.compat.v1.get_default_graph()
    return loaded_model, graph

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('model_resnet.h5')
#print('Model loaded. Check http://127.0.0.1:5000/')
    



def model_predict(img_path, model, graph):
    

    #predicting the image whether it is damaged or not using deep learing trained model.
    
    with graph.as_default():
        img = image.load_img(img_path, target_size=(128,128))

        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)
    
        images = np.vstack([x])
        classes = model.predict(images)
        #print(fn)
        #print(classes)
        if(classes[0][1] >= 0.6):
            #print("Whole Car")
            return "Whole Car"
        else:
            return "Damaged Car"
            #print("Damaged Car")
            
            
            
def model_predictFront(img_path, model, graph):
    

    #predicting the image whether it is damaged or not using deep learing trained model.
    
    with graph.as_default():
        img = image.load_img(img_path, target_size=(128,128))

        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)
    
        images = np.vstack([x])
        classes = model.predict(images)
        #print(fn)
        #print(classes)
        if(classes[0][0] >= 0.6):
            return "Front of the Car"
        elif (classes[0][1]):
            return "Rear of the Car"
        else:
            return "Side of the Car"
        

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    

    #preds = model.predict(x)
    #return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    global model, graph
    model, graph = loadDamageModel()
    
    global modelFront, graphFront
    modelFront, graphFront = loadFrontModel()
    
    
    
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        var1 = model_predict(file_path, model, graph)
        var2 = model_predictFront(file_path, modelFront, graphFront)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        preds = var1+" and it is the "+var2
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

