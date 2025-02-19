# Standard Library Imports
import os
import gc
import time
import threading
import tracemalloc

# Third-Party Libraries
import numpy as np
import psutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# TensorFlow and Keras Imports
import tensorflow as tf


# Function to monitor memory and CPU usage
def monitor_resources(mem_usage, cpu_usage, stop_event):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        cpu = process.cpu_percent(interval=None)  # CPU usage percentage
        mem_usage.append(mem)
        cpu_usage.append(cpu)
        time.sleep(0.1)  # Sampling interval

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix with labels.

    Parameters:
    - cm: Confusion matrix.
    - class_names: List of class names.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Function to print evaluation results
def print_evaluation_results(metrics, class_names):
    """
    Print evaluation metrics and display the confusion matrix.

    Parameters:
    - metrics: Dictionary containing evaluation metrics.
    - class_names: List of class names for the confusion matrix.
    """
    print("\n### Evaluation Metrics ###\n")
    print(f"Evaluation Time:       {metrics['inference_time']:.2f} seconds")
    print(f"Peak Memory Usage:     {metrics['peak_memory_usage']:.2f} MB")
    print(f"Average CPU Usage:     {metrics['average_cpu_usage']:.2f} %")
    print(f"Model Size:            {metrics['model_size_mb']:.2f} MB")
    print(f"Accuracy:              {metrics['accuracy']:.3f}")
    print(f"F1 Score:              {metrics['f1']:.3f}")

    # Plot confusion matrix
    print("\n### Confusion Matrix ###\n")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)

# Function to compute evaluation metrics
def compute_metrics(y_test, y_pred):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    return metrics

# Function to calculate model size
def calculate_model_size(model_path):
    model_size_kb = os.path.getsize(model_path) / 1024  # Size in KB
    model_size_mb = model_size_kb / 1024  # Size in MB
    return model_size_mb

def evaluate_quantized_model(tflite_model_path, test_dataset):
    """
    Evaluates a quantized TFLite model using a preprocessed tf.data.Dataset.
    
    Parameters:
    - tflite_model_path: Path to the TFLite model file.
    - test_dataset: A tf.data.Dataset yielding (image, label) pairs.
      Note: The labels can be one-hot encoded; this function will convert them to integers.
      
    Returns:
    - metrics: Dictionary containing evaluation metrics.
    """
    # Set CPU affinity (optional)
    p = psutil.Process(os.getpid())
    p.cpu_affinity([3])
    
    # Resource monitoring lists
    mem_usage = []
    cpu_usage = []

    # Start memory tracking
    tracemalloc.start()
    
    # Event to stop monitoring
    stop_monitoring = threading.Event()
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(mem_usage, cpu_usage, stop_monitoring))
    monitor_thread.start()

    # Start timing
    start_time = time.time()

    # Load and initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    ground_truth = []

    # Iterate over the test dataset (which is preprocessed, batched, and prefetched)
    for batch_images, batch_labels in test_dataset:
        # Convert tensors to NumPy arrays
        batch_images = batch_images.numpy()
        batch_labels = batch_labels.numpy()
        
        for i in range(batch_images.shape[0]):
            image = batch_images[i]
            # Convert one-hot label to integer label (if necessary)
            if batch_labels[i].ndim > 0:
                true_label = np.argmax(batch_labels[i])
            else:
                true_label = batch_labels[i]
            ground_truth.append(true_label)

            # Prepare input data by adding a batch dimension
            input_data = np.expand_dims(image, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output_data)
            predictions.append(pred)

    # Stop timing
    inference_time = time.time() - start_time

    # Stop resource monitoring
    stop_monitoring.set()
    monitor_thread.join()

    # Get peak memory usage (in MB)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak_memory / (1024 * 1024)

    # Compute evaluation metrics using true labels and predictions
    metrics = compute_metrics(ground_truth, predictions)

    # Get model size in MB
    model_size_mb = calculate_model_size(tflite_model_path)

    # Compute CPU usage statistics
    peak_cpu_usage = max(cpu_usage) if cpu_usage else 0
    avg_cpu_usage = np.mean(cpu_usage) if cpu_usage else 0

    # Add all metrics to the dictionary
    metrics.update({
        'inference_time': inference_time,
        'model_size_mb': model_size_mb,
        'peak_cpu_usage': peak_cpu_usage,
        'average_cpu_usage': avg_cpu_usage,
        'peak_memory_usage': peak_memory,
    })

    print_evaluation_results(metrics, 
                             class_names=["Blurry", "Corrupt", "Missing_Data", "Noisy", "Priority"])

    # Clean up
    del predictions, interpreter, mem_usage, cpu_usage, stop_monitoring, monitor_thread
    gc.collect()
    return metrics