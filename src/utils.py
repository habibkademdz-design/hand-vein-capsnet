"""Utility functions for Hand Vein CapsNet"""

import numpy as np
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


def load_config(config_path):
    """Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directories(dirs):
    """Create multiple directories if they don't exist.
    
    Args:
        dirs (list): List of directory paths
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def plot_training_history(history, save_path=None):
    """Plot training history.
    
    Args:
        history (dict): Training history from model.fit()
        save_path (str): Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy')
    axes[0].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history['loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): Class names
        save_path (str): Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_metrics_report(y_true, y_pred, class_names=None):
    """Print classification metrics report.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): Class names
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def normalize_image(img, dtype=np.float32):
    """Normalize image to [0, 1] range.
    
    Args:
        img (array): Input image
        dtype: Output data type
        
    Returns:
        array: Normalized image
    """
    img = img.astype(dtype)
    img_min = img.min()
    img_max = img.max()
    
    if img_max == img_min:
        return np.zeros_like(img, dtype=dtype)
    
    return (img - img_min) / (img_max - img_min)


def resize_image(img, size=(128, 128)):
    """Resize image to specified size.
    
    Args:
        img (array): Input image
        size (tuple): Target size (height, width)
        
    Returns:
        array: Resized image
    """
    from PIL import Image
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)
