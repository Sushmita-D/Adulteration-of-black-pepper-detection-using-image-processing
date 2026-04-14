from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

torch.set_num_threads(1)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

CLASSES = ["others", "papaya", "pepper"]

CLASS_COLORS = {
    "papaya": (0, 0, 255),
    "pepper": (0, 255, 0),
    "others": (0, 165, 255),
}

YOLO_URL = "https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/best.pt"
MOBILENET_URL = "https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/mobilenetv2_best.pth"

YOLO_PATH = MODELS_DIR / "best.pt"
MOBILENET_PATH = MODELS_DIR / "mobilenetv2_best.pth"

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def download_model(url: str, destination: Path):
    if destination.exists():
        return
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)


@lru_cache(maxsize=1)
def load_models():
    download_model(YOLO_URL, YOLO_PATH)
    download_model(MOBILENET_URL, MOBILENET_PATH)

    yolo_model = YOLO(str(YOLO_PATH))
    yolo_model.to("cpu")  # 🔥 force CPU

    classifier = models.mobilenet_v2()
    classifier.classifier[1] = nn.Linear(classifier.last_channel, len(CLASSES))
    classifier.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
    classifier.to(DEVICE)
    classifier.eval()

    return yolo_model, classifier


def run_pipeline(image: Image.Image):
    if image is None:
        return None, {"error": "Upload image"}, pd.DataFrame(), "Total seeds: 0"

    yolo_model, classifier = load_models()

    img = image.convert("RGB")
    img_np = np.array(img)
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 🔥 Resize to reduce memory
    bgr = cv2.resize(bgr, (640, 640))

    results = yolo_model(bgr, conf=0.25, iou=0.45, max_det=100, verbose=False)[0]

    class_counts = {"pepper": 0, "papaya": 0, "others": 0}
    predictions = []

    for i, box in enumerate(results.boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        tensor = tfm(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = classifier(tensor)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(1).item()

        label = CLASSES[pred]
        conf = float(probs[0][pred])

        class_counts[label] += 1

        predictions.append({
            "Seed_ID": i,
            "Class": label,
            "Confidence": round(conf, 3)
        })

        color = CLASS_COLORS[label]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            bgr,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    output = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    df = pd.DataFrame(predictions)

    return output, class_counts, df, f"Total seeds: {len(predictions)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌱 Black Pepper Adulteration Detection")
    gr.Markdown("Detect pepper vs papaya seeds using AI")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Image")
        output_img = gr.Image(label="Result")

    btn = gr.Button("🚀 Analyze")

    with gr.Row():
        counts = gr.JSON(label="Class Distribution")
        total = gr.Textbox(label="Total Seeds")

    table = gr.Dataframe(label="Predictions")

    btn.click(
        fn=run_pipeline,
        inputs=input_img,
        outputs=[output_img, counts, table, total]
    )

if __name__ == "__main__":
    demo.queue().launch()