import os
import uuid
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_from_directory
from torchvision import models, transforms
from ultralytics import YOLO

# Use non-interactive backend
matplotlib.use("Agg")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs_steps"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "crops", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["others", "papaya", "pepper"]

# -------------------------
# DOWNLOAD MODELS FROM HF
# -------------------------
def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

YOLO_URL = "https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/best.pt"
MOBILENET_URL = "https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/mobilenetv2_best.pth"

download_model(YOLO_URL, "best.pt")
download_model(MOBILENET_URL, "mobilenetv2_best.pth")

# -------------------------
# LOAD MODELS
# -------------------------
yolo = YOLO("best.pt")

classifier = models.mobilenet_v2()
classifier.classifier[1] = nn.Linear(classifier.last_channel, len(CLASSES))
classifier.load_state_dict(
    torch.load("mobilenetv2_best.pth", map_location=DEVICE)
)
classifier.to(DEVICE)
classifier.eval()

# -------------------------
# TRANSFORMS
# -------------------------
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

def _save_charts(class_counts, results_list):
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.savefig(OUTPUT_DIR / "step6_class_distribution.png")
    plt.close()

    seed_ids = [r["Seed_ID"] for r in results_list]
    conf_scores = [r["Class_Conf"] for r in results_list]

    plt.figure(figsize=(10, 5))
    plt.bar(seed_ids, conf_scores)
    plt.savefig(OUTPUT_DIR / "step7_confidence_bars.png")
    plt.close()

def run_pipeline(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image")

    results = yolo(img, conf=0.10, iou=0.30, max_det=500)[0]

    crop_info = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        crop_path = OUTPUT_DIR / "crops" / f"seed_{i+1}.jpg"
        cv2.imwrite(str(crop_path), crop)

        crop_info.append((i + 1, crop_path))

    class_counts = {"pepper": 0, "papaya": 0, "others": 0}
    results_list = []

    for seed_id, path in crop_info:
        crop_img = Image.open(path).convert("RGB")
        crop_tensor = tfm(crop_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = classifier(crop_tensor)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(1).item()

        cls_label = CLASSES[pred]
        cls_conf = float(probs[0][pred])
        class_counts[cls_label] += 1

        results_list.append({
            "Seed_ID": seed_id,
            "Class": cls_label,
            "Class_Conf": cls_conf
        })

    _save_charts(class_counts, results_list)

    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_DIR / "step8_results.csv", index=False)

    return {
        "total_seeds": len(results_list),
        "class_counts": class_counts,
        "results": results_list
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/process", methods=["POST"])
def api_process():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    filename = f"{uuid.uuid4().hex}.jpg"
    path = UPLOAD_DIR / filename
    file.save(path)

    result = run_pipeline(path)
    return jsonify(result)

@app.route("/api/outputs/<path:filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/api/download/<filetype>")
def download_file(filetype):
    files = {
        "csv": "step8_results.csv",
    }
    return send_from_directory(OUTPUT_DIR, files[filetype], as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)