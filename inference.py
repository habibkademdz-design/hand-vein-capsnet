"""Inference script for Hand Vein CapsNet"""

import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2

from src.utils import normalize_image, resize_image


class HandVeinInference:
    """Inference pipeline for hand vein recognition.
    
    Args:
        model_path (str): Path to trained model
        img_size (int): Image size (default: 128)
    """
    
    def __init__(self, model_path, img_size=128):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
    
    def preprocess_image(self, img_path):
        """Preprocess image for inference.
        
        Args:
            img_path (str): Path to image
            
        Returns:
            ndarray: Preprocessed image [1, img_size, img_size, 1]
        """
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize
        img = normalize_image(img)
        
        # Add batch and channel dimensions
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, img_path):
        """Predict identity from image.
        
        Args:
            img_path (str): Path to image
            
        Returns:
            tuple: (predicted_class, confidence, all_scores)
        """
        # Preprocess
        img = self.preprocess_image(img_path)
        
        # Predict
        scores = self.model.predict(img, verbose=0)[0]
        
        # Get prediction
        predicted_class = np.argmax(scores)
        confidence = scores[predicted_class]
        
        return predicted_class, confidence, scores
    
    def predict_batch(self, img_dir, pattern='*.png'):
        """Predict identities for batch of images.
        
        Args:
            img_dir (str): Directory with images
            pattern (str): File pattern
            
        Returns:
            list: List of (filename, predicted_class, confidence)
        """
        results = []
        
        img_files = list(Path(img_dir).glob(pattern))
        
        for img_path in img_files:
            try:
                pred_class, confidence, _ = self.predict(str(img_path))
                results.append({
                    'filename': img_path.name,
                    'predicted_class': int(pred_class),
                    'confidence': float(confidence)
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Inference for Hand Vein CapsNet')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--batch_dir', type=str,
                       help='Directory with batch of images')
    parser.add_argument('--pattern', type=str, default='*.png',
                       help='File pattern for batch processing')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = HandVeinInference(args.model)
    
    if args.image:
        # Single image inference
        pred_class, confidence, scores = inference.predict(args.image)
        print(f"Predicted Class: {pred_class}")
        print(f"Confidence: {confidence:.4f}")
        
    elif args.batch_dir:
        # Batch inference
        results = inference.predict_batch(args.batch_dir, args.pattern)
        
        for result in results:
            print(f"{result['filename']}: Class {result['predicted_class']}, "
                  f"Confidence {result['confidence']:.4f}")


if __name__ == '__main__':
    main()
