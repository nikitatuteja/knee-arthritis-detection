# ArthroScan AI - Knee Osteoarthritis Detection & Severity Grading 🦴

An AI-powered medical web application designed to automatically detect and grade Knee Osteoarthritis from standard Knee X-rays using Deep Learning (**DenseNet-121**), **FastAPI**, and a modern web interface.

Developed by **Nikita Tuteja**.

---

## 🌟 Key Features

- **Automated KL Grading**: Classifies knee X-rays into 5 Kellgren-Lawrence grades:
  - `Grade 0: Normal` (Healthy knee joints)
  - `Grade 1: Doubtful` (Minute osteophytes, doubtful joint space narrowing)
  - `Grade 2: Mild` (Definite small osteophytes, intact joint space)
  - `Grade 3: Moderate` (Multiple moderate osteophytes, clear joint space loss)
  - `Grade 4: Severe` (Large osteophytes, severe bone-on-bone contact)
- **High Accuracy AI Engine**: Built on transfer-learned **DenseNet-121** with fine-tuning.
- **Fast Inference**: Analyzes X-ray images in under 3 seconds.
- **Modern User Interface**: Dark-mode glassmorphism design with drag-and-drop file upload, real-time confidence bars, and interactive UI animations.

---

## 🏗️ Architecture

```mermaid
graph LR
    Frontend["🌐 Modern Web UI (HTML/CSS/JS)"] -->|POST /predict| Backend["⚙️ FastAPI Server (Python)"]
    Backend -->|Image Preprocessing (256x256)| Model["🧠 DenseNet-121 Deep Learning Model"]
    Model -->|Severity & Confidence Scores| Backend
    Backend -->|JSON Prediction Output| Frontend
```

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10 or 3.11 installed on your system.

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/nikitatuteja/knee-arthritis-detection.git
cd knee-arthritis-detection

# Create and activate virtual environment
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python backend/main.py
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## 📁 Project Structure

```
├── backend/
│   └── main.py                     # FastAPI application & ML prediction routes
├── models/
│   └── best_model_improved.keras   # Pre-trained DenseNet-121 model weights
├── full code/
│   └── knee_arthritis.py           # Training and experimentation scripts
├── index.html                      # Modern frontend layout
├── script.js                       # Frontend interaction & API client
├── train_final.py                  # Model fine-tuning script
├── train_cv.py                     # 3-Fold cross-validation script
├── train_model.py                  # Custom CNN baseline training script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🌐 Deployment

- **Frontend**: Can be deployed globally using [Vercel](https://vercel.com).
- **Backend**: Can be hosted on [Render](https://render.com) or [Railway](https://railway.app).

---

## 👤 Author
- **Nikita Tuteja** - [GitHub Profile](https://github.com/nikitatuteja)

---

## ⚠️ Disclaimer
*This tool is created for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
