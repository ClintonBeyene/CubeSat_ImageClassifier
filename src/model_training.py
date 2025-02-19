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
from tensorflow_model_optimization.sparsity import keras as pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import prune_scope


def train_and_save(models, train_dataset, val_dataset, test_dataset, batch_size=32, epochs=40):
    """
    Trains a list of pruned Keras models and saves each trained model in multiple formats,
    including TensorFlow SavedModel, HDF5, and Pickle. In addition, the function converts
    the best performing model to TensorFlow Lite format and evaluates its performance.

    The training procedure uses a cosine decay restart learning rate schedule with the Adam
    optimizer and monitors training with several callbacks:
      - UpdatePruningStep and PruningSummaries for pruning updates and logging.
      - EarlyStopping to halt training if the validation loss does not improve.
      - ModelCheckpoint to save the best model based on validation loss.

    After training, the best model is loaded (using the pruning scope), evaluated on the test dataset,
    and then saved in various formats. The function also demonstrates how to reload and evaluate the model
    from each saved format. Finally, it converts the model to TensorFlow Lite format, evaluates it,
    and computes common evaluation metrics such as accuracy, precision, recall, F1 score, and the confusion matrix.

    Parameters
    ----------
    models : list
        A list of Keras models to be trained. Each model is assumed to have pruning applied.
    train_dataset : tf.data.Dataset
        The TensorFlow dataset used for training.
    val_dataset : tf.data.Dataset
        The TensorFlow dataset used for validation.
    test_dataset : tf.data.Dataset
        The TensorFlow dataset used for testing the final model.
    batch_size : int, optional
        The batch size used during training (default is 32).
    epochs : int, optional
        The number of training epochs (default is 40).

    Returns
    -------
    None
        The function saves the trained models in various formats and prints evaluation metrics.
    """
    
    for model in models:

        # Define dataset and batch size
        train_dataset_size = 9711  # Total number of training images
        
        # Calculate the steps per epoch
        steps_per_epoch = train_dataset_size // batch_size
        
        # Define the first decay steps (based on epochs of training)
        first_decay_steps = steps_per_epoch * epochs

        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,  # Start with a higher LR if needed
            first_decay_steps=first_decay_steps,  # Adjust based on dataset size
            t_mul=2.0,  # Multiplier for the cycle length
            m_mul=0.9,  # Reduce max LR by 10% each cycle
            alpha=1e-5  # Minimum LR
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=['accuracy'],
        )

        # Print the model summary.
        model.summary()

        # Define callbacks
        callbacks = [
            pruning_callbacks.UpdatePruningStep(),
            pruning_callbacks.PruningSummaries(log_dir='../model/pruning_logs/'),
            keras.callbacks.EarlyStopping(monitor='val_loss', 
                                          patience=10, mode='auto', 
                                          restore_best_weights=True),
            keras.callbacks.ModelCheckpoint('../model/best_checkpoint_model.h5', 
                            monitor='val_loss', save_best_only=True, 
                            mode='auto', verbose=1)
            
        ]

        # Fit the model using the tf.data.Dataset
        model.fit(
            train_dataset,
            epochs=epochs,
            verbose=1,
            validation_data=val_dataset,
            callbacks=callbacks
        )

        with prune_scope():
            # Load and use the best model saved by ModelCheckpoint
            best_model = keras.models.load_model('../model/best_checkpoint_model.h5')
    
        # Evaluate on test data
        score = best_model.evaluate(test_dataset, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Save as TensorFlow SavedModel
        saved_model_dir = '../model/saved_model/'
        print('Saving model to TensorFlow SavedModel format at:', saved_model_dir)
        keras.models.save_model(best_model, saved_model_dir, save_format='tf')

        # Save as HDF5 (.h5)
        h5_file = '../model/best_model.h5'
        print('Saving model to HDF5 format at:', h5_file)
        model.save(h5_file, save_format='h5')

        # Save as Pickle (.pkl)
        # Since Keras models aren't directly picklable, we store the architecture and weights.
        model_json = best_model.to_json()
        model_weights = best_model.get_weights()
        pkl_file = '../model/model.pkl'
        print('Saving model to Pickle format at:', pkl_file)
        with open(pkl_file, 'wb') as f:
            pickle.dump({'model_json': model_json, 'model_weights': model_weights}, f)

        # Optional: Verify the SavedModel loading
        print('Loading model from TensorFlow SavedModel format...')
        loaded_model = keras.models.load_model(saved_model_dir)
        score = loaded_model.evaluate(test_dataset, verbose=0)
        print('Test loss after loading SavedModel:', score[0])
        print('Test accuracy after loading SavedModel:', score[1])

        
        # Optional: Verify HDF5 model loading
        print('Loading model from HDF5 format...')
        
        with prune_scope():
            loaded_h5_model = keras.models.load_model(h5_file)
            
        score = loaded_h5_model.evaluate(test_dataset, verbose=0)
        print('Test loss after loading HDF5 model:', score[0])
        print('Test accuracy after loading HDF5 model:', score[1])
        
        
        # Optional: Verify Pickle model loading
        print('Loading model from Pickle format...')
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        with prune_scope():
            loaded_pickle_model = keras.models.model_from_json(data['model_json'])
            
        loaded_pickle_model.set_weights(data['model_weights'])
        # Compile the loaded pickle model before evaluating.
        loaded_pickle_model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy'],
        )
        score = loaded_pickle_model.evaluate(test_dataset, verbose=0)
        
        print('Test loss after loading Pickle model:', score[0])
        print('Test accuracy after loading Pickle model:', score[1])

        # Convert and save as TFLite (.tflite)
        print('Converting the best model to TFLite format...')
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_file = '../model/model.tflite'
        with open(tflite_file, 'wb') as f:
            f.write(tflite_model)
        print('TFLite model saved at:', tflite_file)

        # Load and evaluate the TFLite (.tflite) model 
        print('Loading and evaluating the TFLite model...')
        interpreter = tf.lite.Interpreter(model_path=tflite_file)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Variables to store predictions and ground truth.
        predictions = []
        ground_truth = []
    
        for batch_images, batch_labels in test_dataset:
            batch_images_np = batch_images.numpy()
            batch_labels_np = batch_labels.numpy()
            for i in range(len(batch_images_np)):
                # Process ground truth label (handle one-hot encoding if needed)
                label = batch_labels_np[i]
                if label.ndim > 0:
                    label = np.argmax(label)
                ground_truth.append(label)
                
                # Prepare input and run inference on the TFLite model.
                input_data = np.expand_dims(batch_images_np[i], axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                pred = np.argmax(output_data)
                predictions.append(pred)
                
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

