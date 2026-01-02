from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load YOLO model once
model = YOLO("model/best.pt")

@app.route("/", methods=["GET"])
def home():
    return "YOLO Plastic vs Metal API is running", 200

@app.route("/debug", methods=["GET"])
def debug():
    return "DEBUG ROUTE ACTIVE", 200

@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "image missing"}), 400

    try:
        img_bytes = request.files["image"].read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    result = model.predict(img, imgsz=224, verbose=False)[0]
    probs = result.probs.data.tolist()

    plastic_prob = probs[model.names.index("plastic")]
    metal_prob = probs[model.names.index("metal")]

    if plastic_prob >= metal_prob:
        final_class = "plastic"
        confidence = plastic_prob
    else:
        final_class = "metal"
        confidence = metal_prob

    return jsonify({
        "label": final_class,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
