from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload

# --------------------------------------------------
# Load YOLO model ONCE at startup
# --------------------------------------------------
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

print("Model task:", model.task)
print("Model classes:", model.names)

# Hard safety check
if model.task != "classify":
    raise RuntimeError("ERROR: This server expects a CLASSIFICATION model, not detection.")

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "YOLO Plastic vs Metal API (Classification Mode)", 200

# --------------------------------------------------
# Image loader (file or base64)
# --------------------------------------------------
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

# --------------------------------------------------
# Main inference endpoint
# --------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    img = load_image(request)

    if img is None:
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "reason": "invalid image"
        })

    # --------------------------------------------------
    # YOLO inference (CLASSIFICATION MODE)
    # --------------------------------------------------
    try:
        result = model.predict(img, verbose=False)[0]
    except Exception:
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "reason": "model inference failure"
        })

    probs = result.probs

    if probs is None:
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "reason": "no probabilities returned"
        })

    names = model.names

    # Get correct index of each class
    try:
        metal_idx = list(names.values()).index("metal")
        plastic_idx = list(names.values()).index("plastic")
    except ValueError:
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "reason": "class names mismatch in model"
        })

    metal_prob = float(probs[metal_idx])
    plastic_prob = float(probs[plastic_idx])

    # --------------------------------------------------
    # PURE DECISION LOGIC (NO BIAS)
    # --------------------------------------------------
    if metal_prob > plastic_prob:
        label = "metal"
        confidence = metal_prob
    else:
        label = "plastic"
        confidence = plastic_prob

    # --------------------------------------------------
    # Final response
    # --------------------------------------------------
    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "metal_prob": round(metal_prob, 4),
        "plastic_prob": round(plastic_prob, 4)
    })

# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
