"""
    Referensi:
    - https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
    - https://github.com/avinassh/pytorch-flask-api
"""

import io
import json

from flask import Flask, render_template, request, redirect, jsonify

from inference import get_prediction
from commons import format_class_name




app = Flask(__name__)

@app.route('/vehicle-checker', methods=['GET'])
def vehicle-checker():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'hello':'test success'})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))