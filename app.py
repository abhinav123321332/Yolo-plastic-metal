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

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# --------------------------------------------------
# Load YOLO model ONCE
# --------------------------------------------------
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

# --------------------------------------------------
# Sanity check (important)
# --------------------------------------------------
# Uncomment once to verify:
# print("Model task:", model.task)
# print("Model classes:", model.names)

# --------------------------------------------------
# Class alias mapping (VERY IMPORTANT)
# --------------------------------------------------
METAL_ALIASES = {
    "metal",
    "aluminium",
    "aluminum",
    "can",
    "tin",
    "metal_can",
    "crushed_metal",
    "alu"
}

PLASTIC_ALIASES = {
    "plastic",
    "plastic_bottle",
    "wrapper",
    "polybag"
}

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "YOLO Plastic vs Metal API (Detection-based, hardened)", 200


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
            "reason": "invalid or missing image"
        })

    # --------------------------------------------------
    # YOLO inference (DETECTION MODE)
    # --------------------------------------------------
    try:
        result = model.predict(
            img,
            imgsz=416,       # higher resolution helps crushed objects
            conf=0.15,       # allow weak detections
            iou=0.45,
            verbose=False
        )[0]
    except Exception:
        return jsonify({
            "label": "plastic",
            "confidence": 0.5,
            "reason": "model inference failure"
        })

    boxes = result.boxes

    # --------------------------------------------------
    # No detections → uncertainty handling
    # --------------------------------------------------
    if boxes is None or len(boxes) == 0:
        return jsonify({
            "label": "metal",
            "confidence": 0.55,
            "reason": "no detections; metal-biased uncertainty fallback"
        })

    # --------------------------------------------------
    # Aggregate confidence by material
    # --------------------------------------------------
    names = {k: v.lower() for k, v in result.names.items()}

    scores = {
        "metal": 0.0,
        "plastic": 0.0
    }

    detections_detail = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = names.get(cls_id, "unknown")

        detections_detail.append({
            "class": cls_name,
            "confidence": round(conf, 4)
        })

        if cls_name in METAL_ALIASES:
            scores["metal"] += conf
        elif cls_name in PLASTIC_ALIASES:
            scores["plastic"] += conf
        # unknown classes are ignored intentionally

    metal_score = scores["metal"]
    plastic_score = scores["plastic"]
    total_score = metal_score + plastic_score + 1e-6

    # --------------------------------------------------
    # Decision logic (explicit & honest)
    # --------------------------------------------------
    if metal_score > plastic_score:
        label = "metal"
        confidence = metal_score / total_score
        reason = "metal detections dominate"

    elif plastic_score > metal_score:
        label = "plastic"
        confidence = plastic_score / total_score
        reason = "plastic detections dominate"

    else:
        # tie or extremely weak evidence
        label = "metal"
        confidence = 0.55
        reason = "tie / low evidence; metal-biased resolution"

    # --------------------------------------------------
    # Final response (transparent)
    # --------------------------------------------------
    return jsonify({
        "label": label,
        "confidence": round(confidence, 4),
        "metal_score": round(metal_score, 4),
        "plastic_score": round(plastic_score, 4),
        "detections": len(boxes),
        "details": detections_detail,
        "reason": reason
    })


# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
