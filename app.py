import pickle
from flask import Flask, request, app, render_template, jsonify, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
resmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(list(data.values()))  
    print(np.array(list(data.values())).reshape(1, -1)) 
    new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = resmodel.predict(new_data)
    print(output[0])
    return jsonify({'prediction': output[0]})
    

if __name__ == '__main__':
    app.run(debug=True)