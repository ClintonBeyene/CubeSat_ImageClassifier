# Third-Party Libraries
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# TensorFlow and Keras Imports
import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras


# TensorFlow Model Optimization (TF-MOT) Imports
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.quantization.keras import quantize_scope
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras.collab_opts.prune_preserve import (
    default_8bit_prune_preserve_quantize_scheme,
)


def prune_preserve_quantize_model(saved_model_path, train_dataset, val_dataset, 
                                  test_dataset, batch_size, epochs, train_dataset_size):
    """
    Loads a pruned model, applies quantization, fine-tunes it with training data, and exports
    the resulting quantization-aware model in various formats (SavedModel, HDF5, Pickle, TFLite).
    
    The function follows these steps:
    
    1. **Load and Strip Pruning:**
       - Loads a pruned model from the provided `saved_model_path` within a custom quantization scope
         (which registers the custom pruning wrapper).
       - Removes the pruning wrappers using `prune.strip_pruning()`.
       
    2. **Quantization Annotation and Application:**
       - Annotates the pruned model for quantization using `quantize.quantize_annotate_model()`.
       - Applies quantization (using a default 8-bit prune-preserve quantization scheme) via 
         `quantize.quantize_apply()`.
       - Prints the model summary of the resulting quantization-aware model.
       
    3. **Quantization-Aware Fine-Tuning:**
       - Compiles and fine-tunes the quantization-aware model on the provided `train_dataset` and `val_dataset`.
       - Uses callbacks including:
         - Early stopping (monitors validation loss with a patience of 5 epochs and restores the best weights).
         - Model checkpointing (saves the best model based on validation loss).
       - The training is performed using the helper function `compile_and_fit()`, which should be defined 
         elsewhere in your code.
       
    4. **Model Evaluation and Export:**
       - Loads the best model saved during fine-tuning (within a quantization scope) and evaluates it on 
         `test_dataset`.
       - Saves the best model in three formats:
         - TensorFlow SavedModel format.
         - HDF5 format.
         - Pickle format (saves the model architecture as JSON and its weights).
       - Optionally verifies the SavedModel and HDF5 formats by reloading and evaluating them on the test set.
       
    5. **TFLite Conversion and Evaluation:**
       - Converts the best model to TFLite format with default optimizations.
       - Saves the TFLite model to disk.
       - Loads and evaluates the TFLite model on the test dataset:
         - For each sample, inference is performed using the TFLite interpreter.
         - Computes evaluation metrics such as accuracy, precision, recall, F1 score, and the confusion matrix.
         
    6. **Return:**
       - Returns the best quantization-aware model after fine-tuning.
    
    Parameters:
    -----------
    saved_model_path : str
        File path to the saved pruned model (with pruning wrappers) in TensorFlow format.
    train_dataset : tf.data.Dataset
        The training dataset used for fine-tuning the quantization-aware model.
    val_dataset : tf.data.Dataset
        The validation dataset for monitoring training progress and early stopping.
    test_dataset : tf.data.Dataset
        The test dataset used for final evaluation of the quantized model.
    batch_size : int
        Batch size for training and evaluation.
    epochs : int
        Number of training epochs for fine-tuning.
    train_dataset_size : int
        The total number of samples in the training dataset (used by the training function).
    
    Returns:
    --------
    best_model : tf.keras.Model
        The best quantization-aware model obtained after fine-tuning, evaluated on the test dataset.
    """
    
    # Load the pruned model (wrapped with custom pruning scope)
    with quantize_scope({'PruneLowMagnitude': prune.pruning_wrapper.PruneLowMagnitude}):
        pruned_model = keras.models.load_model(saved_model_path)
    
    # Remove pruning wrappers before quantization.
    pruned_model = prune.strip_pruning(pruned_model)
    
    # Annotate and apply quantization.
    quant_aware_annotate_model = quantize.quantize_annotate_model(pruned_model)
    quant_aware_model = quantize.quantize_apply(
        quant_aware_annotate_model,
        scheme=default_8bit_prune_preserve_quantize_scheme.Default8BitPrunePreserveQuantizeScheme()
    )
    
    quant_aware_model.summary()

    # Define callbacks for training, including early stopping.
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', 
                                      patience=5, mode='auto', 
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('../model/quantized/best_checkpoint_model.h5', 
                                        monitor='val_loss', save_best_only=True, 
                                        mode='auto', verbose=1)
    ]
    
    fit_kwargs = {
        'batch_size': batch_size,
        'epochs': epochs,
        'callbacks': callbacks
    }
    
    compile_and_fit(
        quant_aware_model,
        train_dataset,
        val_dataset,
        batch_size, 
        epochs,
        train_dataset_size,
        compile_kwargs={},  # Additional compile settings if needed.
        fit_kwargs=fit_kwargs
    )

    # Load the best model saved during training.
    with quantize_scope({'QuantizeWrapper': quantize.quantize_wrapper.QuantizeWrapper}):
        best_model = keras.models.load_model('../model/quantized/best_checkpoint_model.h5')
    
    # Evaluate on test data.
    score = best_model.evaluate(test_dataset, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # Save the best model in various formats.
    saved_model_dir = '../model/quantized/saved_model/'
    print('Saving model to TensorFlow SavedModel format at:', saved_model_dir)
    keras.models.save_model(best_model, saved_model_dir, save_format='tf')

    h5_file = '../model/quantized/best_model.h5'
    print('Saving model to HDF5 format at:', h5_file)
    best_model.save(h5_file, save_format='h5')

    pkl_file = '../model/quantized/model.pkl'
    print('Saving model to Pickle format at:', pkl_file)
    model_json = best_model.to_json()
    model_weights = best_model.get_weights()
    with open(pkl_file, 'wb') as f:
        pickle.dump({'model_json': model_json, 'model_weights': model_weights}, f)

    # Optionally verify SavedModel and HDF5 formats by reloading and evaluating.
    print('Loading model from TensorFlow SavedModel format...')
    loaded_model = keras.models.load_model(saved_model_dir)
    score = loaded_model.evaluate(test_dataset, verbose=0)
    print('Test loss after loading SavedModel:', score[0])
    print('Test accuracy after loading SavedModel:', score[1])
    
    print('Loading model from HDF5 format...')
    with quantize_scope({'QuantizeWrapper': quantize.quantize_wrapper.QuantizeWrapper}):
        loaded_h5_model = keras.models.load_model(h5_file)
    score = loaded_h5_model.evaluate(test_dataset, verbose=0)
    print('Test loss after loading HDF5 model:', score[0])
    print('Test accuracy after loading HDF5 model:', score[1])
    
    # Convert and save as TFLite.
    print('Converting the best model to TFLite format...')
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_file = '../model/quantized/model.tflite'
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    print('TFLite model saved at:', tflite_file)

    # Evaluate the TFLite model.
    print('Loading and evaluating the TFLite model...')
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    ground_truth = []
    
    for batch_images, batch_labels in test_dataset:
        batch_images_np = batch_images.numpy()
        batch_labels_np = batch_labels.numpy()
        for i in range(len(batch_images_np)):
            # Process ground truth label.
            label = batch_labels_np[i]
            if label.ndim > 0:
                label = np.argmax(label)
            ground_truth.append(label)
            
            # Run inference on the TFLite model.
            input_data = np.expand_dims(batch_images_np[i], axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(np.argmax(output_data))
    
    # Compute evaluation metrics.
    tflite_accuracy = accuracy_score(ground_truth, predictions)
    tflite_precision = precision_score(ground_truth, predictions, average='weighted')
    tflite_recall = recall_score(ground_truth, predictions, average='weighted')
    tflite_f1 = f1_score(ground_truth, predictions, average='weighted')
    tflite_cm = confusion_matrix(ground_truth, predictions)
    
    print('TFLite Model Evaluation:')
    print('Accuracy:', tflite_accuracy)
    print('Precision:', tflite_precision)
    print('Recall:', tflite_recall)
    print('F1 Score:', tflite_f1)
    print('Confusion Matrix:\n', tflite_cm)
    
    return best_model


# Compile_and_fit function with NaN stabilization.
def compile_and_fit(model, train_dataset, val_dataset, batch_size, epochs, train_dataset_size=9711, compile_kwargs={}, fit_kwargs={}):
    """
    Compiles and trains a Keras model with built-in stabilization to prevent NaN values during training.

    This function configures the model with a cosine decay restart learning rate schedule using a reduced initial learning rate 
    to improve numerical stability. It also applies gradient clipping via both `clipvalue` and `clipnorm` to control the magnitude 
    of gradient updates. In addition, a custom callback is used to monitor the training process for any occurrence of NaN values in 
    the loss or metric outputs; if detected, training is halted immediately.

    Parameters
    ----------
    model : keras.Model
        The Keras model to compile and train.
    train_dataset : tf.data.Dataset
        The TensorFlow Dataset to be used for training.
    val_dataset : tf.data.Dataset
        The TensorFlow Dataset to be used for validation.
    batch_size : int
        The number of samples per training batch.
    epochs : int
        The total number of training epochs.
    train_dataset_size : int, optional
        The total number of training samples (default is 9711). This is used to calculate the number of steps per epoch.
    compile_kwargs : dict, optional
        Additional keyword arguments to pass to the model's `compile` method.
    fit_kwargs : dict, optional
        Additional keyword arguments to pass to the model's `fit` method.

    Returns
    -------
    keras.Model
        The trained Keras model.

    Notes
    -----
    - The learning rate schedule is defined using `CosineDecayRestarts` with a lower initial learning rate (1e-4) for stability.
    - Both `clipvalue` and `clipnorm` are applied in the Adam optimizer to help mitigate the risk of exploding gradients.
    - The custom `NanDetectionCallback` is added to monitor for NaN values in the training logs; if any NaNs are detected in any batch, 
      training is immediately halted.
    """
    # Calculate the steps per epoch.
    steps_per_epoch = train_dataset_size // batch_size
    first_decay_steps = steps_per_epoch * epochs

    # Lower the initial learning rate for stability.
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,  # Lowered from 1e-3.
        first_decay_steps=first_decay_steps,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )

    # Use both clipvalue and clipnorm to control gradient updates.
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipvalue=1.0,
        clipnorm=1.0
    )

    compile_args = {
        'optimizer': optimizer,
        'loss': keras.losses.categorical_crossentropy,
        'metrics': ['accuracy'],
    }
    compile_args.update(compile_kwargs)
    model.compile(**compile_args)

    # Custom callback to monitor for NaNs in training.
    class NanDetectionCallback(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs:
                for key, value in logs.items():
                    if np.any(np.isnan(value)):
                        print(f"NaN detected in {key} at batch {batch}, stopping training.")
                        self.model.stop_training = True

    # Combine any existing callbacks with our NaN detection callback.
    callbacks = fit_kwargs.get('callbacks', [])
    callbacks.append(NanDetectionCallback())
    fit_kwargs['callbacks'] = callbacks

    # Prepare fit arguments.
    fit_args = {
        'validation_data': val_dataset,
        'batch_size': batch_size,
        'epochs': epochs
    }
    fit_args.update(fit_kwargs)

    model.fit(train_dataset, **fit_args)
    
    return model 


