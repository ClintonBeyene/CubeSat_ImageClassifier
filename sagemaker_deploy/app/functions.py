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

