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
# Debug route (deployment check)
# -----------------------------
@app.route("/debug", methods=["GET"])
def debug():
    return "DEBUG ROUTE ACTIVE", 200

# -----------------------------
# Frontend API endpoint
# -----------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    # 1. Check file
    if "image" not in request.files:
        return jsonify({"error": "image missing"}), 400

    # 2. Read image safely
    try:
        img_bytes = request.files["image"].read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    # 3. YOLO inference
    result = model.predict(img, imgsz=224, verbose=False)[0]

    # 4. Extract probabilities and class names
    probs = result.probs.data.tolist()   # list of probabilities
    names = result.names                 # dict: {index: "plastic"/"metal"}

    plastic_index = None
    metal_index = None

    for idx, name in names.items():
        if name == "plastic":
            plastic_index = idx
        elif name == "metal":
            metal_index = idx

    if plastic_index is None or metal_index is None:
        return jsonify({"error": "Class labels not found"}), 500

    plastic_prob = probs[plastic_index]
    metal_prob = probs[metal_index]

    # 5. Decide final class
    if plastic_prob >= metal_prob:
        final_class = "plastic"
        confidence = plastic_prob
    else:
        final_class = "metal"
        confidence = metal_prob

    # 6. Response
    return jsonify({
        "label": final_class,
        "confidence": round(float(confidence), 4)
    })

# -----------------------------
# Run server (Render compatible)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
