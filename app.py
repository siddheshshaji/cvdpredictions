import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('cardio_2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = list([float(x) for x in request.form.values()])
    if model.predict([features])==[1]:
        return render_template('home.html', prediction_text="You most probably have a cardiovascular disease... Please get yourself checked by a physician.")
    else:
        return render_template('home.html', prediction_text="You most probably don't have a cardiovascular disease.")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)