import pandas as pd
import sklearn
from flask import Flask, render_template, request
import numpy as np
import joblib
from utils import *


app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        lotarea = float(request.form['lot_area'])
        lot_frontage = float(request.form['lot_frontage'])
        total_room = int(request.form['total_room'])
        # y = B1 * lotarea + B2 * lot_frontage + B3*total_room + B0
        # y = [B1, B2, B3]*[lotarea lot_frontage total_room] + B0
        [b1, b2, b3], b0 = build_model()
        prediction = b1*lotarea + b2*lot_frontage + b3*total_room
        # print(prediction)
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

