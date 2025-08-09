import numpy as np
import tensorflow as tf
import os
from io import BytesIO
from PIL import Image

# TFLite model path
TFLITE_MODEL_PATH = "Cubenet_Best_Quantized.tflite"

# Class names for model
CLASS_NAMES = ["Blurry", "Corrupt", "Missing_Data", "Noisy", "Priority"]

_cached_input_shape = None

def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    global _cached_input_shape
    if _cached_input_shape is None:
        _cached_input_shape = tuple(interpreter.get_input_details()[0]['shape'])  # (1,H,W,3)
        print(f"[INIT] Model input shape detected: {_cached_input_shape}")
    return interpreter

def _infer_required_size():
    if _cached_input_shape is not None and len(_cached_input_shape) == 4:
        return int(_cached_input_shape[1]), int(_cached_input_shape[2])
    return 128, 128

def preprocess_image(image_bytes, target_size=(128, 128)):
    # Training pipeline (from notebook): (raw_uint8 -> /255.0 -> tf.image.resize -> float32)
    req_h, req_w = _infer_required_size()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)  # shape (H0,W0,3) in 0..255
    arr /= 255.0  # scale first (matches notebook order)
    # Use tf.image.resize (bilinear default) to match training exactly
    arr = tf.image.resize(arr, (req_h, req_w), method='bilinear').numpy()
    # Sanity checks
    if arr.min() < -1e-3 or arr.max() > 1.0005:
        print(f"[WARN] Value range after preprocess unexpected: min={arr.min()} max={arr.max()}")
    arr = np.expand_dims(arr.astype(np.float32, copy=False), axis=0)  # (1,H,W,3)
    if arr.shape[1:3] != (req_h, req_w):
        raise ValueError(f"Preprocessed shape {arr.shape[1:3]} != expected {(req_h, req_w)}")
    print(f"[PREP] shape={arr.shape} range=({arr.min():.4f},{arr.max():.4f})")
    return arr
