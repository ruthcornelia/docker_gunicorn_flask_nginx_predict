from flask import Flask, request
import gunicorn
import pickle #untuk ambil model
import pandas as pd
import numpy as np
import os

path = os.path.dirname(os.path.realpath(__file__))
path = path.replace("\\", "/")

with open(path + '/rf.pkl','rb') as model_file:
     model = pickle.load(model_file)

def create_app():
    app = Flask(__name__)
    @app.route('/predict')
    def predict_iris():
        """Example endpoint returning a prediction of iris
        ---
        parameters:
        - name: s_length
        in: query
        type: number
        required: true
        - name: s_width
        in: query
        type: number
        required: true
        - name: p_length
        in: query
        type: number
        required: true
        - name: p_width
        in: query
        type: number
        required: true
        responses:
        200: 
            description: ok
        """
        s_length = request.args.get("s_length")
        s_width = request.args.get("s_width")
        p_length = request.args.get("p_length")
        p_width = request.args.get("p_width")
        prediction = model.predict(np.array([[s_length,s_width,p_length,p_width]]))
        return str(prediction)

    @app.route('/predict_file', methods = ["POST"])
    def predict_iris_csv():
        """Example file endpoint returning a prediction of iris
        ---
        parameters:
        - name: input_file
        in: formData
        type: file
        required: true
        responses:
        200: 
            description: ok
        """
        input_data = pd.read_csv(request.files.get("input_file"))
        prediction = model.predict(input_data)
        return str(list(prediction))
    return app