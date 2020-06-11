# IMPORTING THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import logging
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

# INITIALIZING SENTRY(FOR LOGGING)
sentry_sdk.init(
    dsn="https://52c6e8b6f80c4a2f9ced545c629958c9@o401877.ingest.sentry.io/5262168",
    integrations=[FlaskIntegration()]
)

# INITIALIZING THE APP
app = Flask(__name__)

# IMPORTING THE MODEL
model = pickle.load(open('cardio_2.pkl', 'rb'))

# LANDING PAGE
@app.route('/')
def home():
    return render_template('home.html')

# 404 PAGE HANDLER
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

# PREDICTION ROUTE
@app.route('/predict',methods=['POST'])
def predict():
    features = list([int(x) for x in request.form.values()]) # THE MODEL NEEDS AN INPUT ARRAY 
    if model.predict([features])==[1]:
        return render_template('home.html', prediction_text="You most probably have a cardiovascular disease... Please get yourself checked by a physician.")
    else:
        return render_template('home.html', prediction_text="It's highly unlikely that you have a cardiovascular disease.")

if __name__ == "__main__":
    app.run()
