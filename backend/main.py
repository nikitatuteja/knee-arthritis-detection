
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the root directory
# (Note: Frontend is now at root)
app.mount("/static", StaticFiles(directory="."), name="static")

# Load the model
def load_best_model():
    # Priority order for models
    possible_paths = [
        "models/best_model_improved.keras", # The "absolutely fabulous" working model (256x256)
        "models/best_model.keras",          # Previous version (224x224)
        "models/arthritis_model.h5"         # Legacy model (128x128)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Map paths to their respective input sizes
            if "best_model_improved" in path:
                size = 256
            elif "best_model" in path:
                size = 224
            elif "arthritis_model.h5" in path:
                size = 128
            else:
                size = 256 # Default fallback
            
            print(f"Loading model from {path} with input size {size}")
            try:
                return tf.keras.models.load_model(path), size
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return None, 256

model, IMG_SIZE = load_best_model()

# Classes
CLASS_NAMES = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/script.js")
async def get_script():
    return FileResponse("script.js")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read and preprocess the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    result = {
        "class": CLASS_NAMES[predicted_class_idx],
        "confidence": confidence,
        "all_predictions": {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    }
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
