import pickle
from flask import Flask, request, render_template, jsonify
import numpy as np

# Load model and scaler
with open("regmodel.pkl", "rb") as f:
    resmodel = pickle.load(f)

with open("scaling.pkl", "rb") as f:
    scalar = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    
    if 'data' not in data:
        return jsonify({"error": "Invalid input format"}), 400

    try:
        values = list(data['data'].values())
        input_array = np.array(values).reshape(1, -1)
        transformed_data = scalar.transform(input_array)
        output = resmodel.predict(transformed_data)
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    prediction = resmodel.predict(final_input)[0]
    
    return render_template('home.html', prediction_text=f"{prediction}")

if __name__ == '__main__':
    app.run(debug=True)
