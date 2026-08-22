import os
import sys
import io
import base64
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocess import apply_clahe, CLEAN_CLASS_NAMES
from src.gradcam import get_gradcam_heatmap, generate_superimposed_gradcam, encode_image_to_base64

app = FastAPI(
    title="ArthroScan AI API",
    description="Automated Knee Osteoarthritis Grading with Grad-CAM Explainable AI",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount reports and static assets
reports_dir = os.path.join(ROOT_DIR, "reports")
if os.path.exists(reports_dir):
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

# Load model
def load_best_model():
    possible_paths = [
        os.path.join(ROOT_DIR, "models", "best_model_improved.keras"), # Active high-accuracy DenseNet-121 model (256x256)
        os.path.join(ROOT_DIR, "models", "best_model.keras"),          # MobileNetV2 (224x224)
        os.path.join(ROOT_DIR, "models", "arthritis_model.h5")         # Legacy model (128x128)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            size = 256 if "best_model_improved" in path else (224 if "best_model" in path else 128)
            print(f"[MODEL] Loading model from {path} with input size {size}...")
            try:
                loaded = tf.keras.models.load_model(path)
                print("[MODEL] Model loaded successfully into memory.")
                return loaded, size
            except Exception as e:
                print(f"[MODEL ERROR] Error loading {path}: {e}")
    return None, 256

model, IMG_SIZE = load_best_model()

@app.get("/")
async def root():
    index_path = os.path.join(ROOT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "healthy", "service": "ArthroScan AI API"})

@app.get("/script.js")
async def get_script():
    script_path = os.path.join(ROOT_DIR, "script.js")
    if os.path.exists(script_path):
        return FileResponse(script_path)
    raise HTTPException(status_code=404, detail="script.js not found")

@app.get("/health")
async def health():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "input_size": IMG_SIZE,
        "classes": CLEAN_CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, IMG_SIZE
    if model is None:
        model, IMG_SIZE = load_best_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Train the model first.")

    # Read and decode image
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format. Please upload PNG, JPG, or JPEG.")

    raw_np = np.array(pil_img)
    raw_resized = cv2.resize(raw_np, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE preprocessing & normalization
    enhanced_np = apply_clahe(raw_resized) / 255.0
    input_tensor = np.expand_dims(enhanced_np, axis=0)

    # Model inference
    predictions = model.predict(input_tensor, verbose=0)
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    # Generate Grad-CAM Explainable AI Heatmap
    gradcam_base64 = None
    try:
        heatmap = get_gradcam_heatmap(input_tensor, model, pred_index=predicted_class_idx)
        superimposed = generate_superimposed_gradcam(raw_resized, heatmap, alpha=0.45)
        gradcam_base64 = encode_image_to_base64(superimposed)
    except Exception as e:
        print(f"[Grad-CAM Notice] Heatmap generation fallback: {e}")

    result = {
        "class": CLEAN_CLASS_NAMES[predicted_class_idx],
        "confidence": confidence,
        "all_predictions": {
            CLEAN_CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLEAN_CLASS_NAMES))
        },
        "gradcam_image": f"data:image/jpeg;base64,{gradcam_base64}" if gradcam_base64 else None
    }
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
