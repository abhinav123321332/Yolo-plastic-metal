from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
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

# -------------------------------
# Image loader
# -------------------------------
def load_image(req):
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

# -------------------------------
# Classification endpoint
# -------------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    img = load_image(request)

    if img is None:
        return jsonify({
            "label": "plastic",
            "confidence": 0.50,
            "reason": "invalid or missing image"
        }), 200

    try:
        result = model.predict(img, verbose=False)[0]
    except Exception as e:
        return jsonify({
            "label": "plastic",
            "confidence": 0.50,
            "reason": f"model inference failure: {str(e)}"
        }), 200

    # -------------------------------
    # Proper classification handling
    # -------------------------------
    probs = result.probs

    if probs is None:
        return jsonify({
            "label": "plastic",
            "confidence": 0.50,
            "reason": "no probabilities returned"
        }), 200

    prob_tensor = probs.data  # torch tensor [metal, plastic]

    # Safety check
    if len(prob_tensor) != 2:
        return jsonify({
            "label": "plastic",
            "confidence": 0.50,
            "reason": "model is not binary classifier"
        }), 200

    metal_prob = float(prob_tensor[0].item())
    plastic_prob = float(prob_tensor[1].item())

    # -------------------------------
    # Final decision (forced binary)
    # -------------------------------
    if metal_prob > plastic_prob:
        label = "metal"
        confidence = metal_prob
    else:
        label = "plastic"
        confidence = plastic_prob

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "metal_prob": round(metal_prob, 4),
        "plastic_prob": round(plastic_prob, 4)
    }), 200

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0", port=port)
