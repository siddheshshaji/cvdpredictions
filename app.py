import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import logging
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="https://52c6e8b6f80c4a2f9ced545c629958c9@o401877.ingest.sentry.io/5262168",
    integrations=[FlaskIntegration()]
)

app = Flask(__name__)
model = pickle.load(open('cardio_2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

@app.route('/predict',methods=['POST'])

def predict():
    features = list([float(x) for x in request.form.values()])
    if model.predict([features])==[1]:
        return render_template('home.html', prediction_text="You most probably have a cardiovascular disease... Please get yourself checked by a physician.")
    else:
        return render_template('home.html', prediction_text="You most probably don't have a cardiovascular disease.")

if __name__ == "__main__":
    app.run()
