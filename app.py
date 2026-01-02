from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

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
    return "YOLO Plastic vs Metal API is running", 200

# -----------------------------
# DEBUG route (deployment check)
# -----------------------------
@app.route("/debug", methods=["GET"])
def debug():
    return "DEBUG ROUTE ACTIVE", 200

# -----------------------------
# API Upload route (FRONTEND USES THIS)
# -----------------------------
@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    # YOLO inference
    results = model(image)[0]

    probs = results.probs
    class_names = results.names

    best_index = probs.top1
    label = class_names[best_index]
    confidence = float(probs.top1conf)

    # Enforce only plastic / metal
    if label not in ["plastic", "metal"]:
        label = "plastic" if label.lower().startswith("plast") else "metal"

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4)
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
