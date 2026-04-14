# 🌱 AI-Based Black Pepper Adulteration Detection

An end-to-end deep learning system to detect and classify **black pepper and adulterants (papaya seeds)** using computer vision.

---

## 🚀 Live Demo

👉 https://huggingface.co/spaces/sushmitadaivajna/black-pepper-detection

---

## 🧠 Problem Statement

Food adulteration is a major issue in agricultural supply chains.
Black pepper is often mixed with **papaya seeds**, which look visually similar.

This project uses **deep learning + image processing** to automatically detect and classify seeds, enabling fast and accurate quality inspection.

---

## ⚙️ Tech Stack

* **YOLOv8** → Object Detection
* **MobileNetV2** → Classification
* **PyTorch** → Model training & inference
* **OpenCV** → Image processing
* **Gradio** → Web UI
* **Hugging Face Spaces** → Deployment

---

## 🔄 Pipeline Overview

1. Upload image 📷
2. YOLO detects individual seeds
3. Each seed is cropped
4. MobileNet classifies:

   * Pepper
   * Papaya
   * Others
5. Final annotated output is displayed

---

## 📊 Features

* 🖼️ Image upload & processing
* 🔍 Real-time seed detection
* 🧠 Classification (pepper vs adulterants)
* 📈 Class distribution output
* 📋 Prediction table
* 🌐 Deployed live app

---

## 📦 Model Weights

Models are hosted on Hugging Face:

* YOLO Model:
  https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/best.pt

* MobileNet Model:
  https://huggingface.co/sushmitadaivajna/balck-pepper-detection/resolve/main/mobilenetv2_best.pth

👉 Models are automatically downloaded when the app runs.

---

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/black-pepper-detection.git
cd black-pepper-detection

pip install -r requirements.txt
python app.py
```

---

## 📁 Project Structure

```
black-pepper-detection/
│── app.py
│── requirements.txt
│── templates/
│── seed_yolovm/
```

---

## 🌍 Real-World Applications

* Food quality inspection
* Agricultural labs
* Supply chain monitoring
* Adulteration detection in spices

---

## 🔥 Future Improvements

* Adulteration percentage calculation
* Mobile app integration
* Real-time camera detection
* Multi-seed classification

---

## 👩‍💻 Author

**Sushmita D**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share!
