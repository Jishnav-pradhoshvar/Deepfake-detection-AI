from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from pathlib import Path

app = Flask(__name__)
CORS(app)
model = load_model("model/deepfake_model.h5")
BASE_DIR = Path(__file__).resolve().parent

@app.route('/', methods=['GET'])
def home():
    return send_from_directory(BASE_DIR, "deepguard_premium_ui.html")

@app.route('/deepguard', methods=['GET'])
def deepguard():
    return send_from_directory(BASE_DIR, "deepguard_premium_ui.html")

@app.route('/legacy', methods=['GET'])
def legacy():
    return send_from_directory(BASE_DIR, "deepfake_detector_ui.html")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "Deepfake detection API is running.",
        "predict_endpoint": "POST /predict"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({ "error": "Missing image file" }), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({ "error": "Invalid image data" }), 400

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    score = float(model.predict(img, verbose=0)[0][0])
    return jsonify({ "fake_score": score })

if __name__ == '__main__':
    app.run(port=5000)
