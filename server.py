'''
Developed by Tomas Medeot, February 2023
Email: tomimedeot@gmail.com
'''

import sys
import os
import json

import flask
import tensorflow as tf
from neural_network import predict

def _load_model():
    global model
    model = tf.keras.models.load_model('Trained_Model')
    
app = flask.Flask(__name__)#Set the app server

#Routes
@app.route('/',methods=['GET','POST'])
def index():
    if flask.request.method == 'GET':
        return None

    elif flask.request.method == 'POST':
        '''
        Save the image for use as example on future trains and then predict
        '''
        elemens = os.listdir('DataSet/Server_Predict/')
        if elemens:
            name = int(max(elemens).rsplit('.', 1)[0])+1
        else:
            name = 0
        img = flask.request.files['img']
        img.save(os.path.join('DataSet/Server_Predict/'+str(name)+'.png'))
        prediction=predict(model,DATA['CATEGORIES'],'DataSet/Server_Predict/',str(name)+'.png')
        return {'prediction':prediction}

    else:
        return None

if __name__ == '__main__':
    try:
        with open('data.json', 'r') as file:
            DATA = json.load(file)
        _load_model()
    except:
        sys.exit('Failed loading elements')
    app.run(DATA['HOST'],DATA['PORT'])
