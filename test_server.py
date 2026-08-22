
print("Starting test script...")
import sys
sys.path.insert(0, '.')
print("Importing main...")
try:
    from backend.main import app, model, IMG_SIZE
    print("Successfully imported main")
    print(f"Model loaded: {model is not None}, IMG_SIZE: {IMG_SIZE}")
    import uvicorn
    print("About to run uvicorn...")
    uvicorn.run(app, host='0.0.0.0', port=8000)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    print("Traceback:")
    traceback.print_exc()

