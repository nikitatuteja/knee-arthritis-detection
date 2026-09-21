import os
import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI(title="Knee Arthritis Analyzer")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Model
MODEL_PATH = "model.h5"
try:
    # Try to load the model if it exists
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Warning: Could not load model.h5. Did you run knee_arthritis.py yet? Error: {e}")

# The categories based on your dataset structure
CATEGORIES = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
IMG_SIZE = 256

def generate_medical_report(prediction_label: str) -> str:
    """Calls local Ollama to generate a report based on the CNN prediction."""
    prompt = f"You are an expert radiologist. A Convolutional Neural Network has analyzed a patient's knee X-ray and predicted the arthritis severity as: '{prediction_label}'. Write a brief, professional medical report summarizing what this means for the patient and provide some generic recommendations for this level of severity. Keep it concise, professional, and well-structured with bullet points for recommendations."
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "Error: No response from LLM.")
    except Exception as e:
        return f"Could not generate report from Ollama. Ensure Ollama is running. Details: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="The CNN model (model.h5) is not loaded. Please train it first by running knee_arthritis.py.")
    
    # Read the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    # Preprocess image for the model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    data = np.array(resized) / 255.0
    data = data.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    # Predict
    prediction = model.predict(data)
    class_idx = np.argmax(prediction[0])
    severity = CATEGORIES[class_idx]
    confidence = float(np.max(prediction[0]))
    
    # Generate Report
    report = generate_medical_report(severity)
    
    return {
        "severity": severity,
        "confidence": confidence,
        "report": report
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
