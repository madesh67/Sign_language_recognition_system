<div align="center">

# 🤟 Sign Language Recognition System

Recognize hand gestures and translate them into text using **Python, OpenCV, MediaPipe, and Machine Learning**.  
Empowering communication for everyone with the power of AI.

---

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-red?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Integrated-orange?logo=google&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Used-yellow?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Custom-purple?logo=open-source-initiative&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![MadeWithLove](https://img.shields.io/badge/Built%20with-❤️-pink)

</div>

---

## 🎯 Overview

The **Sign Language Recognition System** is designed to recognize various hand gestures and translate them into corresponding letters or words.  
This project utilizes **MediaPipe** for hand tracking and **Scikit-Learn** for training a static classifier model.

---

## ⚙️ Features

- 🖐️ Capture new hand signs with labels for training
- 🧠 Train a classification model on captured data
- 📷 Real-time hand sign recognition via webcam
- 💾 Save and reuse trained models

---

## 🧩 Installation

```bash
# Clone the repository
git clone https://github.com/madesh67/Sign_language_recognition_system.git

# Navigate to the project directory
cd Sign_language_recognition_system

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

**Required Packages:**
```
opencv-python
mediapipe
numpy
scikit-learn
joblib
```

---

## 🧠 Usage

### 1️⃣ Capture New Hand Signs

Used to collect and label hand sign samples for training.

```bash
python capture.py --labels "A,B,C" --output data/samples.csv --samples-per-label 100
```

**Arguments:**

| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--labels-json` | str | "" | Path to labels.json |
| `--labels` | str | "" | Comma-separated labels |
| `--output` | str | "data/samples.csv" | Output CSV file |
| `--samples-per-label` | int | 100 | Target samples per label |
| `--camera` | int | 0 | Camera index |
| `--mirror` | flag | False | Mirror preview |
| `--min-det` | float | 0.6 | Minimum detection confidence |
| `--min-trk` | float | 0.6 | Minimum tracking confidence |
| `--save-references` | flag | False | Save reference images |
| `--reference-dir` | str | "reference_signs" | Directory for reference images |
| `--ref-interval` | int | 10 | Save reference every N captures |

---

### 2️⃣ Train the Model

Used to train a classifier model from the captured samples.

```bash
python train.py
```

After training, a model named **`isl_static_clf.joblib`** will be saved automatically.

---

### 3️⃣ Run the Application

Recognize hand signs in real-time using your webcam.

```bash
python app.py
```

No command-line arguments are required — the webcam starts automatically and displays detected gestures.

---

## 🧰 Project Structure

The repository includes scripts for capturing data, training models, and running real-time recognition.  
Refer to the GitHub directory view for the complete folder structure.

---

## 💡 How It Works

1. **Capture Stage** – Collects labeled hand landmarks using MediaPipe.  
2. **Training Stage** – Trains a Scikit-Learn classifier using the captured data.  
3. **Recognition Stage** – Loads the trained model and recognizes gestures in real-time.  

---

## 🧾 License

This project is released under a **Custom License**.  
You are free to use and modify it for educational and non-commercial purposes with proper credit to the author.

---

## 📬 Contact

👤 **Author:** Madesh  
📧 **Email:** madesh1367@gmail.com  
🌐 **GitHub:** [madesh67](https://github.com/madesh67)

---

<div align="center">

✨ *“Bridging communication barriers through AI and technology.”* ✨

</div>
