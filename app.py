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

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
CLASSES = ["others", "papaya", "pepper"]
CLASS_COLORS = {
    "papaya": (0, 0, 255),   # Red in BGR
    "pepper": (0, 255, 0),   # Green in BGR
    "others": (0, 165, 255), # Orange in BGR
}

YOLO_URL = "https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/best.pt"
MOBILENET_URL = "https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/mobilenetv2_best.pth"
YOLO_PATH = MODELS_DIR / "best.pt"
MOBILENET_PATH = MODELS_DIR / "mobilenetv2_best.pth"

tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def download_model(url: str, destination: Path) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(destination, "wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_obj.write(chunk)


@lru_cache(maxsize=1)
def load_models():
    download_model(YOLO_URL, YOLO_PATH)
    download_model(MOBILENET_URL, MOBILENET_PATH)

    yolo_model = YOLO(str(YOLO_PATH))
    cls_model = models.mobilenet_v2()
    cls_model.classifier[1] = nn.Linear(cls_model.last_channel, len(CLASSES))
    cls_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
    cls_model.to(DEVICE)
    cls_model.eval()
    return yolo_model, cls_model


def run_pipeline(image: Image.Image):
    if image is None:
        return None, {"error": "Please upload an image first."}, pd.DataFrame(), "Total seeds: 0"

    yolo_model, cls_model = load_models()

    rgb_img = image.convert("RGB")
    img_np = np.array(rgb_img)
    bgr_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    detections = yolo_model(bgr_img, conf=0.10, iou=0.30, max_det=500, verbose=False)[0]
    boxes = detections.boxes

    class_counts = {"pepper": 0, "papaya": 0, "others": 0}
    predictions = []

    for seed_id, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(bgr_img.shape[1], x2)
        y2 = min(bgr_img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop_bgr = bgr_img[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crop_tensor = tfm(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = cls_model(crop_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()

        cls_label = CLASSES[pred_idx]
        cls_conf = float(probs[0][pred_idx])
        class_counts[cls_label] += 1

        predictions.append(
            {
                "Seed_ID": seed_id,
                "Class": cls_label,
                "Class_Conf": round(cls_conf, 4),
                "Box": f"[{x1}, {y1}, {x2}, {y2}]",
            }
        )

        label_text = f"{cls_label} ({cls_conf:.2f})"
        box_color = CLASS_COLORS.get(cls_label, (255, 255, 255))
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            bgr_img,
            label_text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )

    output_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    table_df = pd.DataFrame(predictions)
    total_text = f"Total seeds: {len(predictions)}"
    return output_rgb, class_counts, table_df, total_text


with gr.Blocks(title="Adulterated Pepper Seed Detector & Classifier") as demo:
    gr.Markdown("# Adulterated Pepper Seed Detector & Classifier")
    gr.Markdown("Run YOLO detection, classify each seed, and open each output in its own page.")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload an image")
        annotated_image = gr.Image(type="numpy", label="Annotated Final")

    process_btn = gr.Button("🚀 Process Image", variant="primary")

    with gr.Row():
        class_counts_out = gr.JSON(label="Class Distribution")
        total_count_out = gr.Textbox(label="Total Count", interactive=False)

    predictions_table = gr.Dataframe(
        headers=["Seed_ID", "Class", "Class_Conf", "Box"],
        datatype=["number", "str", "number", "str"],
        label="Predictions Table",
        interactive=False,
    )

    process_btn.click(
        fn=run_pipeline,
        inputs=input_image,
        outputs=[annotated_image, class_counts_out, predictions_table, total_count_out],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)