import base64
import io
import json
from PIL import Image
from ultralytics import YOLO
import paho.mqtt.client as mqtt

MODEL = YOLO("model/best.pt")

MQTT_BROKER = "82cbd8d556e34122b459b88d24462a9.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "server"
MQTT_PASS = "Server@123"

IMAGE_TOPIC = "wastesort/image"
RESULT_TOPIC = "wastesort/result"

def on_message(client, userdata, msg):
    try:
        img_bytes = base64.b64decode(msg.payload)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        result = MODEL(img)[0]
        label = result.names[result.probs.top1]
        confidence = float(result.probs.top1conf)

        payload = {
            "label": label,
            "confidence": round(confidence, 4)
        }

        client.publish(RESULT_TOPIC, json.dumps(payload))
        print("Published result:", payload)

    except Exception as e:
        print("MQTT processing error:", e)

client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.tls_set()
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT)
client.subscribe(IMAGE_TOPIC)

print("MQTT listener running...")
client.loop_forever()
