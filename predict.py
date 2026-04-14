import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model/deepfake_model.h5")

# ── IMPORTANT: verify this matches train_data.class_indices ─────────────────
# If class_indices was {'fake': 0, 'real': 1}:
#   prediction > 0.5  →  REAL
#   prediction < 0.5  →  FAKE
CLASS_MAP = {0: "FAKE", 1: "REAL"}

def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MobileNetV2 expects RGB
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_input, verbose=0)[0][0]
    label = CLASS_MAP[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Image  : {img_path}")
    print(f"Result : {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw score : {prediction:.4f}  (>0.5 = REAL, <0.5 = FAKE)")
    return label, confidence

predict_image("test.jpg")