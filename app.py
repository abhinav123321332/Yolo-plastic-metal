from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2
import torch

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# -------------------------------
# Load classification model
# -------------------------------
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

print("Model task:", model.task)
print("Model classes:", model.names)

# Must be exactly these two
METAL_LABEL = "metal"
PLASTIC_LABEL = "plastic"

# -------------------------------
# Health check
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return "YOLO Plastic vs Metal Classifier API", 200

# ============================================================
# IMAGE LOADERS
# ============================================================

# Browser-style uploads
def load_image_browser(req):
    try:
        if "image" in req.files:
            return Image.open(
                io.BytesIO(req.files["image"].read())
            ).convert("RGB")

        if req.is_json and "image" in req.json:
            return Image.open(
                io.BytesIO(base64.b64decode(req.json["image"]))
            ).convert("RGB")
    except Exception:
        return None

    return None


# ESP32 raw JPEG uploads
def load_image_esp32(req):
    try:
        if req.data is None or len(req.data) == 0:
            return None

        img_array = np.frombuffer(req.data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    except Exception as e:
        print("ESP32 IMAGE DECODE ERROR:", e)
        return None


# ============================================================
# SHARED CLASSIFICATION LOGIC
# ============================================================

def classify_image(img: Image.Image):

    try:
        result = model.predict(img, verbose=False)[0]
    except Exception as e:
        return {
            "label": "plastic",
            "confidence": 0.50,
            "reason": f"model inference failure: {str(e)}"
        }

    probs = result.probs

    if probs is None:
        return {
            "label": "plastic",
            "confidence": 0.50,
            "reason": "no probabilities returned"
        }

    prob_tensor = probs.data  # torch tensor [metal, plastic]

    if len(prob_tensor) != 2:
        return {
            "label": "plastic",
            "confidence": 0.50,
            "reason": "model is not binary classifier"
        }

    metal_prob = float(prob_tensor[0].item())
    plastic_prob = float(prob_tensor[1].item())

    if metal_prob > plastic_prob:
        label = "metal"
        confidence = metal_prob
    else:
        label = "plastic"
        confidence = plastic_prob

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "metal_prob": round(metal_prob, 4),
        "plastic_prob": round(plastic_prob, 4)
    }


# ============================================================
# BROWSER API (unchanged)
# ============================================================

@app.route("/api/upload", methods=["POST"])
def upload():
    img = load_image_browser(request)

    if img is None:
        return jsonify({
            "label": "plastic",
            "confidence": 0.50,
            "reason": "invalid or missing image"
        }), 200

    result = classify_image(img)
    return jsonify(result), 200


# ============================================================
# ESP32-CAM API (NEW + REQUIRED)
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():

    img = load_image_esp32(request)

    if img is None:
        return jsonify({
            "label": "plastic",
            "confidence": 0.50,
            "reason": "invalid or missing esp32 image"
        }), 400

    result = classify_image(img)
    return jsonify(result), 200


# ============================================================
# Run server
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
