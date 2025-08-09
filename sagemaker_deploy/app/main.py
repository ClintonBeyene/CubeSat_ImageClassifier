

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from .functions import load_tflite_model, preprocess_image, predict_class, CLASS_NAMES


app = FastAPI(title="CubeSat Image Classification API")

# Allow CORS for local dev, Vercel prod, and Vercel preview deployments
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://cube-sat-image-classifier.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model at startup


@app.post("/predict", summary="Classify an uploaded image")
async def classify_image(file: UploadFile = File(...)):
    import os
    import uuid
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        contents = await file.read()
        # Save uploaded image for debugging
        debug_dir = "debug_uploads"
        os.makedirs(debug_dir, exist_ok=True)
        fname = f"{uuid.uuid4().hex}_{file.filename}"
        fpath = os.path.join(debug_dir, fname)
        with open(fpath, "wb") as f:
            f.write(contents)

        image = preprocess_image(contents)
        # Debug: print array stats
        print(f"[DEBUG] {file.filename} shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
        # Reload interpreter for each prediction to avoid stale state
        interpreter = load_tflite_model()
        pred_idx, prob = predict_class(interpreter, image)
        # Debug: print model output
        print(f"[DEBUG] Model output for {file.filename}: class={CLASS_NAMES[pred_idx]}, prob={prob}")
        return JSONResponse({
            "predicted_class": CLASS_NAMES[pred_idx],
            "class_index": int(pred_idx),
            "probability": float(prob)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
