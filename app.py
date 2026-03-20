from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# -------------------------------
# Load classification model
# -------------------------------
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

print("✅ Model loaded:", MODEL_PATH)
print("Model task:", model.task)
print("Model classes:", model.names)

# -------------------------------
# Material class mappings
# -------------------------------
PLASTIC_CLASSES = {
    "plastic",
    "plastic bottle",
    "water_bottle",
    "pet",
    "plastic_bag",
    "cup",
    "syringe",
    "tray",
    "wine_bottle",
    "polyethylene",
    "wrapper",
    "packet",
    "oxygen_mask",
    "sleeping_bag",
    "nipple",
    "bag"
}

METAL_CLASSES = {
    "metal",
    "can",
    "oil_filter",
    "aluminium",
    "crash_helmet",
    "spotlight",
    "steel",
    "hard_disc",
    "bobsled",
    "beer_bottle",
    "loudspeaker",
    "pop_bottle",
    "lighter",
    "sarong",
    "coffee_mug",
    "tin"
}

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
            "label": None,
            "confidence": 0.00,
            "reason": f"model inference failure: {str(e)}"
        }

    probs = result.probs

    if probs is None:
        return {
            "label": None,
            "confidence": 0.00,
            "reason": "no probabilities returned"
        }

    best_idx = int(probs.top1)
    raw_label = result.names[best_idx].lower()
    confidence = round(float(probs.top1conf), 4)

    if raw_label in PLASTIC_CLASSES:
        material = "plastic"
    elif raw_label in METAL_CLASSES:
        material = "metal"
    else:
        print(f"❌ Unknown label: {raw_label}")
        return {
            "label": None,
            "confidence": confidence,
            "raw_label": raw_label,
            "reason": "image is neither metal nor plastic"
        }

    print(f"🧠 Raw: {raw_label} → FINAL: {material} ({confidence})")

    return {
        "label": material,
        "confidence": confidence,
        "raw_label": raw_label
    }


# ============================================================
# BROWSER API
# ============================================================

@app.route("/api/upload", methods=["POST"])
def upload():
    img = load_image_browser(request)

    if img is None:
        return jsonify({
            "label": None,
            "confidence": 0.00,
            "reason": "invalid or missing image"
        }), 400

    result = classify_image(img)
    status = 200 if result["label"] is not None else 422
    return jsonify(result), status


# ============================================================
# ESP32-CAM API
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():

    img = load_image_esp32(request)

    if img is None:
        return jsonify({
            "label": None,
            "confidence": 0.00,
            "reason": "invalid or missing esp32 image"
        }), 400

    result = classify_image(img)
    status = 200 if result["label"] is not None else 422
    return jsonify(result), status


# ============================================================
# Run server
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)