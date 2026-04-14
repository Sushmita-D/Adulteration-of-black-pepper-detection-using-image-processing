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
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_from_directory
from torchvision import models, transforms
from ultralytics import YOLO


# Use non-interactive backend for servers
matplotlib.use("Agg")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs_steps"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "crops", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["others", "papaya", "pepper"]

# -------------------------
# Load Models at Startup
# -------------------------
yolo = YOLO(BASE_DIR / "seed_yolovm" / "weights" / "best.pt")

classifier = models.mobilenet_v2()
classifier.classifier[1] = nn.Linear(classifier.last_channel, len(CLASSES))
classifier.load_state_dict(
    torch.load(
        BASE_DIR / "mobilenetv2_best.pth",
        map_location=DEVICE,
        weights_only=True,
    )
)
classifier.to(DEVICE)
classifier.eval()

tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


def _save_charts(class_counts, results_list):
    """Generate pie and bar charts."""
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Class Distribution of Seeds")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step6_class_distribution.png")
    plt.close()

    seed_ids = [r["Seed_ID"] for r in results_list]
    conf_scores = [r["Class_Conf"] for r in results_list]

    plt.figure(figsize=(10, 5))
    plt.bar(seed_ids, conf_scores)
    plt.xlabel("Seed ID")
    plt.ylabel("MobileNet Confidence")
    plt.title("Confidence per Classified Seed")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step7_confidence_bars.png")
    plt.close()


def run_pipeline(image_path: Path):
    """Runs detection + classification and saves all outputs."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image")

    cv2.imwrite(str(OUTPUT_DIR / "step1_original.jpg"), img)

    results = yolo(img, conf=0.10, iou=0.30, max_det=500)[0]
    yolo_img = results.plot()
    cv2.imwrite(str(OUTPUT_DIR / "step2_yolo_detection.jpg"), yolo_img)

    crop_info = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_yolo = float(box.conf[0])

        crop = img[y1:y2, x1:x2]
        crop_path = OUTPUT_DIR / "crops" / f"seed_{i+1}.jpg"
        cv2.imwrite(str(crop_path), crop)

        crop_info.append((i + 1, crop_path, (x1, y1, x2, y2), conf_yolo))

    class_counts = {"pepper": 0, "papaya": 0, "others": 0}
    results_list = []

    for seed_id, path, coords, yolo_conf in crop_info:
        crop_img = Image.open(path).convert("RGB")
        crop_tensor = tfm(crop_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = classifier(crop_tensor)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(1).item()

        cls_label = CLASSES[pred]
        cls_conf = float(probs[0][pred])
        class_counts[cls_label] += 1

        results_list.append(
            {
                "Seed_ID": seed_id,
                "Class": cls_label,
                "Class_Conf": cls_conf,
                "YOLO_Conf": yolo_conf,
                "x1": coords[0],
                "y1": coords[1],
                "x2": coords[2],
                "y2": coords[3],
            }
        )

    with open(OUTPUT_DIR / "step4_classification.txt", "w", encoding="utf-8") as f:
        for r in results_list:
            f.write(
                f"Seed {r['Seed_ID']} → {r['Class']} "
                f"(MobileNet={r['Class_Conf']:.2f}, YOLO={r['YOLO_Conf']:.2f})\n"
            )

        f.write("\nCLASS COUNTS:\n")
        for c in class_counts:
            f.write(f"{c}: {class_counts[c]}\n")

        f.write(f"\nTotal Seeds: {len(results_list)}\n")

    final_img = img.copy()
    for r in results_list:
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        label = f"{r['Class']} {r['Class_Conf']:.2f}"
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            final_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    cv2.imwrite(str(OUTPUT_DIR / "step5_final_output.jpg"), final_img)

    _save_charts(class_counts, results_list)

    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_DIR / "step8_results.csv", index=False)

    return {
        "total_seeds": len(results_list),
        "class_counts": class_counts,
        "outputs": {
            "original": "step1_original.jpg",
            "detection": "step2_yolo_detection.jpg",
            "classification_txt": "step4_classification.txt",
            "final": "step5_final_output.jpg",
            "pie": "step6_class_distribution.png",
            "bar": "step7_confidence_bars.png",
            "csv": "step8_results.csv",
        },
        "crops": [str(path.name) for _, path, _, _ in crop_info],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/page/<path:filename>")
def view_page(filename):
    return render_template("view.html", filename=filename)


@app.route("/api/process", methods=["POST"])
def api_process():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Only JPG/PNG files are supported"}), 400

    save_name = f"upload_{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / save_name
    file.save(save_path)

    try:
        summary = run_pipeline(save_path)
    except Exception as e:  # pylint: disable=broad-except
        return jsonify({"error": str(e)}), 500

    return jsonify(summary)


@app.route("/api/outputs/<path:filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/api/download/<filetype>")
def download_file(filetype):
    files = {
        "csv": "step8_results.csv",
        "txt": "step4_classification.txt",
    }
    if filetype not in files:
        return jsonify({"error": "Invalid file type"}), 400
    return send_from_directory(OUTPUT_DIR, files[filetype], as_attachment=True)


if __name__ == "__main__":
    # Disable Flask reloader to avoid repeated model loads and "file not found" loops on Windows.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

