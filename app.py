from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
<<<<<<< HEAD
import io
import base64
import numpy as np
import cv2
import torch
=======
import requests
import os, uuid, traceback, time

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

UPLOAD_DIR = "uploads"
MODEL_PATH = "yolov8n-cls.pt"

TELEGRAM_BOT_TOKEN = "8154196275:AAFSj5f4PB1jIXuzLESzIftHQ1ONNCugB8o"
TELEGRAM_CHAT_ID = "1815073816"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------
# MATERIAL MAPPING (STRICT, NO FALLBACK)
# -------------------------------------------------

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
    "pet",
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

# -------------------------------------------------
# APP
# -------------------------------------------------
>>>>>>> db83188 (server updated)

app = Flask(__name__)
CORS(app)

<<<<<<< HEAD
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# -------------------------------
# Load classification model
# -------------------------------
MODEL_PATH = "model/best.pt"
=======
@app.before_request
def preflight():
    if request.method == "OPTIONS":
        return "", 200
    print(f"🔌 {request.method} {request.path} | IP: {request.remote_addr}")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

print("🔄 Loading YOLO classification model...")
>>>>>>> db83188 (server updated)
model = YOLO(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)

# -------------------------------------------------
# HEALTH
# -------------------------------------------------

<<<<<<< HEAD
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
=======
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# -------------------------------------------------
# SERVE IMAGES
# -------------------------------------------------

@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# -------------------------------------------------
# UPLOAD ENDPOINT
# -------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload():
    start = time.time()

    try:
        image_file = request.files.get("image")
        machine = request.form.get("machineName")

        if not image_file or not machine:
            return jsonify({
                "success": False,
                "error": "image and machineName are required"
            }), 400

        # Save image
        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(UPLOAD_DIR, filename)
        image_file.save(image_path)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Inference
        result = model(image, verbose=False)[0]

        probs = result.probs.data.tolist()
        names = result.names
        best_idx = int(result.probs.top1)

        raw_label = names[best_idx].lower()
        confidence = round(float(probs[best_idx]), 3)

        # -------------------------------------------------
        # STRICT MATERIAL DECISION (NO GUESSING)
        # -------------------------------------------------

        if raw_label in PLASTIC_CLASSES:
            material = "plastic"
        elif raw_label in METAL_CLASSES:
            material = "metal"
        else:
            print(f"❌ Unknown material: {raw_label}")
            return jsonify({
                "success": False,
                "error": "Image is neither metal nor plastic"
            }), 422

        print(f"🧠 Raw: {raw_label} → FINAL: {material} ({confidence})")
        print(f"⏱️ Time: {round(time.time() - start, 3)}s")

        send_to_telegram(image_path, material, confidence, machine)

        # -------------------------------------------------
        # ✅ REQUIRED RESPONSE (FIXED)
        # -------------------------------------------------

        return jsonify({
            "success": True,
            "machineName": machine,
            "material": material,
            "label": material,
            "classification": material,
            "confidence": confidence,
            "imageUrl": f"/uploads/{filename}"
        }), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"success": False}), 500

# -------------------------------------------------
# TELEGRAM
# -------------------------------------------------

def send_to_telegram(image_path, material, confidence, machine):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        caption = (
            "♻️ GreenArk Detection\n\n"
            f"Material: {material}\n"
            f"Confidence: {confidence}\n"
            f"Machine: {machine}"
        )

        with open(image_path, "rb") as img:
            requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": img},
                timeout=10
            )
    except Exception as e:
        print("Telegram error:", e)

# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":
    print("🚀 Server running on port 5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
>>>>>>> db83188 (server updated)
