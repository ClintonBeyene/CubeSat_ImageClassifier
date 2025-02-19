# Keras Imports
from tensorflow_model_optimization.python.core.keras.compat import keras


# TensorFlow Model Optimization (TF-MOT) Imports
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.quantization.keras.collab_opts.prune_preserve import (
    default_8bit_prune_preserve_quantize_scheme,
)


def build_layerwise_model(input_shape, num_classes=5, **pruning_params):
    """
    Builds a sequential Keras model with layer-wise pruning for image classification.

    This function constructs a model consisting of multiple convolutional layers 
    (using SeparableConv2D) with pruning applied via low-magnitude pruning, 
    followed by batch normalization, ReLU activation, and max pooling layers. 
    The model then performs global average pooling, applies dropout for regularization, 
    and finally uses a pruned dense layer with softmax activation for classification.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (e.g., (height, width, channels)).
    num_classes : int, optional
        Number of output classes. Default is 5.
    **pruning_params : dict
        Additional keyword arguments for configuring the pruning parameters. 
        These parameters are passed to the `prune.prune_low_magnitude` function for each layer.

    Returns
    -------
    keras.Sequential
        A Keras Sequential model with the defined architecture and pruning applied.
    """
    
    l = keras.layers
    
    return keras.Sequential([
        prune.prune_low_magnitude(
            l.SeparableConv2D(32, (3, 3), padding='same',
                              depthwise_regularizer=keras.regularizers.L2(0.001),
                              pointwise_regularizer=keras.regularizers.L2(0.001)
                             ),
              input_shape=input_shape, **pruning_params),
        l.BatchNormalization(),
        l.Activation('relu'),
        l.MaxPooling2D((2, 2), padding='same'),

        prune.prune_low_magnitude(l.SeparableConv2D(64, (3, 3), padding='same',
                                                    depthwise_regularizer=keras.regularizers.L2(0.001),
                                                    pointwise_regularizer=keras.regularizers.L2(0.001)
                                                   ),
              **pruning_params),
        l.BatchNormalization(),
        l.Activation('relu'),
        l.MaxPooling2D((2, 2), padding='same'),

        prune.prune_low_magnitude(l.SeparableConv2D(128, (3, 3), padding='same',
                                                    depthwise_regularizer=keras.regularizers.L2(0.001),
                                                    pointwise_regularizer=keras.regularizers.L2(0.001)
                                                   ),
              **pruning_params),
        l.BatchNormalization(),
        l.Activation('relu'),
        l.MaxPooling2D((2, 2), padding='same'),

        l.GlobalAveragePooling2D(),
        l.Dropout(0.5),        
        prune.prune_low_magnitude(l.Dense(num_classes), **pruning_params),
        l.Activation('softmax')
    ])

