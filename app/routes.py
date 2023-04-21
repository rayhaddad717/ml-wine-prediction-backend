from flask import jsonify, request
from app.services import predict
from app.utils import WineSample
from app import app
import pytesseract
from PIL import Image
from flask import request, jsonify
from werkzeug.utils import secure_filename
import os
@app.route('/api/convert_image_to_string', methods=['POST'])
def convert_image_to_string():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_name = secure_filename(image.filename)
    image_path = os.path.join('temp', image_name)
    image.save(image_path)

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)

    os.remove(image_path)
    data_dict = {
    'fixed acidity': [6.5],
    'volatile acidity': [0.2],
    'citric acid': [0.35],
    'residual sugar': [1.5],
    'chlorides': [0.05],
    'free sulfur dioxide': [12.0],
    'density': [0.9955],
    'pH': [3.1],
    'sulphates': [0.7],
    'alcohol': [11.5],
    'type': ['red']
    }
    df = WineSample(data_dict).get_dataframe()
    prediction = predict(df)
    return jsonify({"text": text,"prediction":int(prediction)})