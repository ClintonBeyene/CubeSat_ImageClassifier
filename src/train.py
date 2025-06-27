# Import necessary libraries
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay
from src.model_architecture import build_layerwise_model
from src.model_training import train_and_save

# Define the main function to train the model
def main(train_dataset, val_dataset, test_dataset, 
         epochs=40, batch_size=32, total_train_images=9711):
    # Print information about the dataset
    for images, _ in train_dataset.take(1):
        input_shape = images.shape[1:]  # Excluding batch dimension
        break
    print('Input shape:', input_shape)
    print(f"{tf.data.experimental.cardinality(train_dataset)} training batches available.")

    # Set up pruning parameters and build the model
    
    # Calculate pruning steps
    total_images = total_train_images
    batch_size = batch_size
    steps_per_epoch = total_images // batch_size  # Number of batches per epoch
    total_steps = steps_per_epoch * epochs  # epochs of training
    
    begin_step = int(0.1 * total_steps)  # Start pruning at 10% of training
    end_step = int(0.6 * total_steps)  # End pruning at 60% of training
    frequency = total_steps // epochs  # We have to adjust frequency based on training duration
    
    pruning_params = {
        'pruning_schedule': PolynomialDecay(
            initial_sparsity=0.1,
            final_sparsity=0.75,
            begin_step=begin_step,
            end_step=end_step,
            frequency=frequency)
    }

    # Build the model using pruning parameters
    layerwise_model = build_layerwise_model(input_shape, **pruning_params)
    models = [layerwise_model]

    # Train and save the model(s)
    train_and_save(models, 
                   train_dataset, val_dataset, test_dataset)