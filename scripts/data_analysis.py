# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras Imports
import tensorflow as tf


# Unnormalize function: converts standardized images back to their original scale.
def unnormalize(image, mean, var):
    return image * tf.sqrt(var + 1e-8) + mean

# Define the class names
class_names = ["Blurry", "Corrupt", "Missing_Data", "Noisy", "Priority"]

def visualize_image_grid(dataset, mean, var, num_images=32):
    """
    Visualizes a grid of images from a dataset, showing different images each time.

    Parameters:
    - dataset: A tf.data.Dataset yielding (images, labels).
    - mean: Global mean used for unnormalizing the images.
    - var: Global variance used for unnormalizing the images.
    - num_images: Number of images to display.
    """
    # Take one batch from the dataset (this will be different on each run if dataset is shuffled)
    for standardized_images, one_hot_labels in dataset.take(1):
        # Unnormalize images for visualization
        images = unnormalize(standardized_images, mean, var)
        
        # Convert to NumPy arrays. Depending on your original range, you might need to rescale.
        images_np = images.numpy()
        images_np = np.clip(images_np, 0, 255).astype(np.uint8)
        
        # Get the integer labels from one-hot encoding
        labels = tf.argmax(one_hot_labels, axis=-1).numpy()

        # Shuffle the images and labels within the batch to ensure randomness
        indices = np.arange(images_np.shape[0])
        np.random.shuffle(indices)
        
        # Select up to `num_images` images from the shuffled batch
        selected_indices = indices[:num_images]

        # Determine grid size dynamically
        grid_rows = int(np.sqrt(num_images))  # Square root for near-square layout
        grid_cols = int(np.ceil(num_images / grid_rows))  # Ensure enough columns

        # Create a grid and plot images
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 10))
        axes = axes.flatten()
        for i in range(num_images):
            if i < len(selected_indices):
                image_idx = selected_indices[i]
                axes[i].imshow(images_np[image_idx])
                class_label = labels[image_idx]
                class_name = class_names[class_label] if class_label < len(class_names) else "Unknown"
                axes[i].set_title(f"Class: {class_name}")
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        plt.tight_layout()
        plt.show()


def plot_class_distribution(dataset, class_names):
    num_classes = len(class_names)
    class_counts = np.zeros(num_classes)

    # Unbatch the dataset to count each class
    for _, one_hot_label in dataset.unbatch():
        label = tf.argmax(one_hot_label).numpy()
        class_counts[label] += 1

    # Create a Seaborn barplot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_names, y=class_counts, palette="viridis")

    # Add labels and title
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=30)  # Rotate class names for better readability

    # Show values on top of bars
    for i, count in enumerate(class_counts):
        plt.text(i, count + 2, int(count), ha='center', fontsize=12)

    plt.show()


def plot_pixel_histograms(dataset, mean, var, num_batches=1, bins=100, log_scale=False):
    """
    Plots histograms of pixel intensities for each RGB channel
    from a given number of batches in the dataset.

    Parameters:
    - dataset: A tf.data.Dataset that yields (standardized_images, labels).
    - mean: tf.Tensor or NumPy array with shape (3,) containing the global mean.
    - var: tf.Tensor or NumPy array with shape (3,) containing the global variance.
    - num_batches: Number of batches to visualize (default: 1).
    - bins: Number of bins for the histogram (default: 100 for finer granularity).
    - log_scale: Whether to apply log-scaling to the y-axis (default: False).
    """
    for standardized_images, _ in dataset.take(num_batches):
        images = unnormalize(standardized_images, mean, var)
        images_np = np.clip(images.numpy(), 0, 255).astype(np.uint8)

        assert images_np.shape[-1] == 3, "Expected 3 channels (RGB), but got different shape."

        channel_names = ["Red", "Green", "Blue"]
        channel_colors = ["red", "green", "blue"]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        sns.set_style("whitegrid")

        for c in range(3):
            channel_data = images_np[..., c].ravel()
            sns.histplot(channel_data, bins=bins, ax=axs[c], color=channel_colors[c], stat="density")
            axs[c].set_title(f"{channel_names[c]} Channel Pixel Intensity", fontsize=12, fontweight="bold")
            axs[c].set_xlabel("Intensity", fontsize=11)
            axs[c].set_ylabel("Density", fontsize=11)

            if log_scale:
                axs[c].set_yscale("log")

        plt.suptitle("Pixel Intensity Histograms per Channel", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()