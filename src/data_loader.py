"""Data loading and augmentation for Hand Vein dataset"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy.ndimage import map_coordinates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


class HandVeinDataLoader:
    """Data loader for Hand Vein dataset.
    
    Args:
        data_dir (str): Path to data directory containing ROI images
        img_size (int): Target image size (default: 128)
        num_subjects (int): Number of subjects in dataset (default: 276)
    """
    
    def __init__(self, data_dir='data/ROI_Directory/', img_size=128, num_subjects=276):
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_subjects = num_subjects
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.file_names = []
    
    def load_dataset(self):
        """Load all images from dataset directory.
        
        Returns:
            tuple: (images, labels, filenames)
        """
        print(f"Loading images from {self.data_dir}...")
        
        # Get all image files
        image_files = sorted(Path(self.data_dir).glob('*.png')) + \
                     sorted(Path(self.data_dir).glob('*.jpg')) + \
                     sorted(Path(self.data_dir).glob('*.jpeg'))
        
        if not image_files:
            raise ValueError(f"No images found in {self.data_dir}")
        
        print(f"Found {len(image_files)} images")
        
        # Load images
        for img_path in tqdm(image_files, desc="Loading images"):
            # Load image
            img = self._load_image(str(img_path))
            
            # Extract person ID from filename
            # Format: person_XXX_db1_LZ.png
            filename = img_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 2 and parts[0] == 'person':
                person_id = parts[1]
                
                self.images.append(img)
                self.labels.append(person_id)
                self.file_names.append(filename)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        # Encode labels
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        print(f"Loaded {len(self.images)} images")
        print(f"Found {len(np.unique(self.labels))} unique subjects")
        
        return self.images, self.labels, self.file_names
    
    def load_loao_fold(self, fold_num=1):
        """Load Leave-One-Acquisition-Out fold.
        
        Args:
            fold_num (int): Fold number (1, 2, 3, or 4)
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val)
        """
        if fold_num not in [1, 2, 3, 4]:
            raise ValueError("fold_num must be 1, 2, 3, or 4")
        
        self.load_dataset()
        
        # Separate by acquisition
        acquisition_mask = np.array([f'L{fold_num}' in fn for fn in self.file_names])
        
        X_val = self.images[acquisition_mask]
        y_val = self.labels_encoded[acquisition_mask]
        
        X_train = self.images[~acquisition_mask]
        y_train = self.labels_encoded[~acquisition_mask]
        
        print(f"\nLOAO Fold {fold_num}:")
        print(f"  Train: {len(X_train)} images")
        print(f"  Val: {len(X_val)} images")
        
        return X_train, y_train, X_val, y_val
    
    def _load_image(self, img_path):
        """Load and preprocess single image.
        
        Args:
            img_path (str): Path to image
            
        Returns:
            ndarray: Preprocessed image [img_size, img_size, 1]
        """
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        
        return img
    
    def get_augmentation_generator(self, seed=42):
        """Get data augmentation generator.
        
        Returns:
            ImageDataGenerator: Configured data augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
            preprocessing_function=self._elastic_deformation
        )
    
    @staticmethod
    def _elastic_deformation(image, alpha=30, sigma=3):
        """Apply elastic deformation to image.
        
        Args:
            image (ndarray): Input image
            alpha (float): Deformation strength
            sigma (float): Gaussian std
            
        Returns:
            ndarray: Deformed image
        """
        random_state = np.random.RandomState(None)
        
        h, w = image.shape[:2]
        
        # Random displacement fields
        dx = random_state.randn(h, w) * sigma
        dy = random_state.randn(h, w) * sigma
        
        # Smooth fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, sigma=sigma) * alpha
        dy = gaussian_filter(dy, sigma=sigma) * alpha
        
        # Create mesh
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        x_deformed = (x + dx).astype(np.float32)
        y_deformed = (y + dy).astype(np.float32)
        
        # Clip to bounds
        x_deformed = np.clip(x_deformed, 0, w - 1)
        y_deformed = np.clip(y_deformed, 0, h - 1)
        
        # Interpolate
        deformed = map_coordinates(image, [y_deformed, x_deformed], order=1, mode='reflect')
        
        return deformed
    
    def create_tf_dataset(self, X, y, batch_size=32, augment=False, shuffle=True):
        """Create TensorFlow dataset.
        
        Args:
            X (ndarray): Images
            y (ndarray): Labels
            batch_size (int): Batch size
            augment (bool): Apply augmentation
            shuffle (bool): Shuffle dataset
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        if augment:
            # Apply online augmentation
            def augment_fn(x, y):
                x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
                x = tf.image.random_crop(x, size=[self.img_size, self.img_size, 1])
                return x, y
            
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
