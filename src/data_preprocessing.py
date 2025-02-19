# Third-Party Libraries
import numpy as np

# TensorFlow and Keras Imports
import tensorflow as tf


def create_preprocessed_dataset(dataset_path, images_filename, labels_filename, 
                                batch_size=32, num_classes=5, shuffle=False,
                                normalization_params=None):
    """
    Loads images/labels from disk using memory mapping and creates a batched/prefetched tf.data.Dataset.
    Processes data entirely on-demand without loading full arrays into RAM.
    
    Optionally, normalization parameters (global_mean, global_var) can be provided.
    This allows you to compute them once on the training dataset and reuse for validation/test.
    
    Parameters:
        dataset_path (str): Path to the dataset directory.
        images_filename (str): Filename for the images (.npy file).
        labels_filename (str): Filename for the labels (.npy file).
        batch_size (int): Size of the batches.
        num_classes (int): Number of label classes for one-hot encoding.
        shuffle (bool): If True, shuffles the dataset by shuffling indices in the generator.
        normalization_params (tuple or None): A tuple (global_mean, global_var) to be used for normalization.
            If None, computes statistics from the dataset (suitable for training data).
        
    Returns:
        ds (tf.data.Dataset): A dataset yielding preprocessed, batched, and prefetched data.
        normalization_params (tuple): The computed (global_mean, global_var). When providing these for 
                                      validation/test sets, you can ignore the returned values.
    """
    
    # Load memory-mapped arrays (virtual disk pointers)
    images_mmap = np.load(f"{dataset_path}/{images_filename}", mmap_mode='r')
    labels_mmap = np.load(f"{dataset_path}/{labels_filename}", mmap_mode='r')
    
    # Print the shape of the entire dataset (total number of samples)
    print(f"Total samples in {images_filename}: {images_mmap.shape[0]}")
    
    # Compute global normalization parameters only if not provided.
    if normalization_params is None:
        def compute_global_stats(images):
            total_sum = np.zeros(3)
            total_sum_sq = np.zeros(3)
            count = 0
            for i in range(len(images)):
                img = images[i].astype(np.float64)  # use higher precision for accumulation
                total_sum += np.sum(img, axis=(0, 1))  # Sum across height & width, keep channels
                total_sum_sq += np.sum(np.square(img), axis=(0, 1))
                count += img.shape[0] * img.shape[1]  # Total pixels per channel
            mean = total_sum / count
            var = (total_sum_sq / count) - np.square(mean)
            return mean, var
        
        global_mean, global_var = compute_global_stats(images_mmap)
        normalization_params = (global_mean, global_var)
    else:
        global_mean, global_var = normalization_params

    # Convert normalization parameters to tf.constant for use in the pipeline.
    global_mean = tf.constant(global_mean, dtype=tf.float32)
    global_var = tf.constant(global_var, dtype=tf.float32)
    
    # Generator function to read data on demand, with optional shuffling.
    def disk_reader():
        images = images_mmap  # keep reference in scope
        labels = labels_mmap
        indices = np.arange(len(images))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield images[i], labels[i]
    
    # Create dataset with explicit output signature
    output_signature = (
        tf.TensorSpec(shape=images_mmap.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(disk_reader, output_signature=output_signature)
    
    # Preprocessing function using training dataset normalization parameters
    def preprocess_fn(image, label):
        image = tf.cast(image, tf.float32)
        standardized_image = (image - global_mean) / tf.sqrt(global_var + 1e-8)
        one_hot_label = tf.one_hot(label, num_classes)
        return standardized_image, one_hot_label
    
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"Disk-based pipeline created from {dataset_path}")
    
    return ds, normalization_params


def create_full_dataset(dataset_path, images, labels,  
                         num_classes=5, shuffle=False, normalization_params=None):
    """
    Loads the entire dataset into a tf.data.Dataset without loading all data
    into memory at once by using lazy loading with tf.py_function.
    
    Parameters:
        dataset_path (str): Path to the dataset directory.
        images_filename (str): Filename for the images (.npy file).
        labels_filename (str): Filename for the labels (.npy file).
        num_classes (int): Number of label classes for one-hot encoding.
        shuffle (bool): If True, shuffles the dataset.
        normalization_params (tuple or None): A tuple (global_mean, global_var) for normalization.
            If None, computes statistics from the dataset.
    
    Returns:
        dataset (tf.data.Dataset): A dataset yielding all preprocessed images and labels.
        normalization_params (tuple): Computed (global_mean, global_var) if not provided.
    """
    
    # Load memory-mapped arrays (they are not fully loaded into memory)
    # images_mmap = np.load(f"{dataset_path}/{images_filename}", mmap_mode='r')
    # labels_mmap = np.load(f"{dataset_path}/{labels_filename}", mmap_mode='r')
    
    # Compute global normalization parameters if not provided
    if normalization_params is None:
        # Sum over the spatial dimensions (assuming images_mmap.shape = [num_images, height, width, channels])
        total_sum = np.sum(images, axis=(0, 1, 2))
        total_sum_sq = np.sum(np.square(images), axis=(0, 1, 2))
        count = np.prod(images.shape[:3])  # Total number of pixels per channel
        global_mean = total_sum / count
        global_var = (total_sum_sq / count) - np.square(global_mean)
        normalization_params = (global_mean, global_var)
    else:
        global_mean, global_var = normalization_params

    # Convert normalization parameters to TensorFlow constants
    # (We'll need their NumPy values later inside our py_function)
    global_mean_tf = tf.constant(global_mean, dtype=tf.float32)
    global_var_tf = tf.constant(global_var, dtype=tf.float32)
    
    # Define a function that loads and preprocesses one sample.
    # This function will run in eager mode via tf.py_function.
    def load_and_preprocess(index):
        # index comes in as a NumPy array from tf.py_function; convert to int.
        i = int(index)
        image = images[i]           # NumPy array
        label = labels[i]           # NumPy scalar
        
        # Convert image to float32 (if not already)
        image = image.astype(np.float32)
        
        # Use the NumPy values of global_mean and global_var for normalization.
        # We use .numpy() to get the eager (NumPy) value.
        global_mean_np = global_mean_tf.numpy()
        global_var_np = global_var_tf.numpy()
        
        # Normalize the image.
        image = (image - global_mean_np) / np.sqrt(global_var_np + 1e-8)
        
        return image, label

    # Wrap the above function so that it can be used in a tf.data pipeline.
    def map_function(index):
        # tf.py_function runs load_and_preprocess eagerly.
        image, label = tf.py_function(func=load_and_preprocess, inp=[index], 
                                      Tout=[tf.float32, tf.int64])
        # If you know the shape (for example, 512x512x3), set it explicitly:
        image.set_shape([images.shape[1], images.shape[2], images.shape[3]])
        label.set_shape([])  # scalar label
        # One-hot encode the label.
        label = tf.one_hot(label, num_classes)
        return image, label

    # Create a dataset of indices. This avoids loading the full arrays into memory.
    dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(images)))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    # Map the index to actual data using our mapping function.
    dataset = dataset.map(map_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Prefetch for performance.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    print(f"Dataset prepared from {dataset_path}")
    
    return dataset, normalization_params

def compute_and_save_global_stats(dataset_path, images_filename, save_filename):
    """
    Computes the global mean and variance from the training dataset and saves them to a .npy file.
    
    Parameters:
        dataset_path (str): Path to the dataset directory.
        images_filename (str): Filename for the training images (.npy file).
        save_filename (str): Filename to save computed mean and variance (.npy file).
    
    Returns:
        global_mean (np.ndarray): Computed mean.
        global_var (np.ndarray): Computed variance.
    """
    images_mmap = np.load(f"{dataset_path}/{images_filename}", mmap_mode='r')

    total_sum = np.zeros(3)
    total_sum_sq = np.zeros(3)
    count = 0
    
    for i in range(len(images_mmap)):
        img = images_mmap[i].astype(np.float64)  # Higher precision
        total_sum += np.sum(img, axis=(0, 1))
        total_sum_sq += np.sum(np.square(img), axis=(0, 1))
        count += img.shape[0] * img.shape[1]  # Total pixels per channel

    global_mean = total_sum / count
    global_var = (total_sum_sq / count) - np.square(global_mean)

    # Save computed values
    np.save(f"{dataset_path}/{save_filename}", np.array([global_mean, global_var]))
    print(f"Saved normalization parameters to {dataset_path}/{save_filename}")

    return global_mean, global_var