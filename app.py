from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
import base64

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# Limit upload size (5MB)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# -----------------------------
# Load YOLO model ONCE
# -----------------------------
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

# -----------------------------
# Health check
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Plastic vs Metal classifier API running", 200

# -----------------------------
# Debug route
# -----------------------------
@app.route("/debug", methods=["GET"])
def debug():
    return "DEBUG OK", 200

# -----------------------------
# Classification endpoint
# -----------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    # -------------------------
    # 1. Read image (file or base64)
    # -------------------------
    try:
        if "image" in request.files:
            img_bytes = request.files["image"].read()

        elif request.is_json and "image" in request.json:
            img_bytes = base64.b64decode(request.json["image"])

        else:
            return jsonify({
                "label": "plastic",
                "confidence": 0.5,
                "note": "no image received, defaulted"
            })

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    except Exception:
        # ABSOLUTE FALLBACK — NEVER FAIL
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "note": "invalid image, defaulted"
        })

    # -------------------------
    # 2. YOLO inference
    # -------------------------
    try:
        result = model.predict(img, imgsz=224, verbose=False)[0]
    except Exception:
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "note": "model error, defaulted"
        })

    # -------------------------
    # 3. Extract probabilities
    # -------------------------
    if result.probs is None:
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "note": "no probs returned, defaulted"
        })

    probs = result.probs.data.tolist()
    names = {k: v.lower() for k, v in result.names.items()}

    class_probs = {}
    for idx, name in names.items():
        class_probs[name] = float(probs[idx])

    plastic_prob = class_probs.get("plastic", 0.0)
    metal_prob   = class_probs.get("metal", 0.0)

    # -------------------------
    # 4. FORCE binary decision
    # -------------------------
    if plastic_prob >= metal_prob:
        final_class = "plastic"
        confidence = plastic_prob
    else:
        final_class = "metal"
        confidence = metal_prob

    # -------------------------
    # 5. Response (ALWAYS VALID)
    # -------------------------
    return jsonify({
        "label": final_class,
        "confidence": round(confidence, 4),
        "plastic_prob": round(plastic_prob, 4),
        "metal_prob": round(metal_prob, 4)
    })

# -----------------------------
# Run server (Render compatible)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
