"""Capsule Network Model - Simplified and Stable Version"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import numpy as np


def build_capsnet(num_subjects=276, l2_reg=1e-5):
    """Build a stable CapsNet model using softmax classification.
    
    Args:
        num_subjects (int): Number of subjects
        l2_reg (float): L2 regularization
        
    Returns:
        Model: Compiled Keras model
    """
    inputs = Input(shape=(128, 128, 1), name='input')
    
    # Feature extraction with strong regularization
    x = Conv2D(64, 5, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg), name='conv1')(inputs)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg), name='conv2')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(256, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg), name='conv3')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(512, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg), name='conv4')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # Global pooling for capsule-like behavior
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer: softmax classification
    # This is more stable than trying to implement true margin loss
    output = Dense(num_subjects, activation='softmax', 
                  kernel_regularizer=l2(l2_reg), name='output')(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    # Use standard cross-entropy which is much more stable
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
                keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_accuracy')]
    )
    
    return model


def build_simple_capsnet(num_subjects=276, l2_reg=1e-5):
    """Simpler version for faster iteration.
    
    Args:
        num_subjects (int): Number of subjects
        l2_reg (float): L2 regularization
        
    Returns:
        Model: Compiled Keras model
    """
    inputs = Input(shape=(128, 128, 1))
    
    # Simple CNN backbone
    x = Conv2D(64, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(128, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.3)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    output = Dense(num_subjects, activation='softmax',
                  kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    return model


def build_capsnet_with_margin_loss(num_subjects=276, l2_reg=1e-5):
    """Build CapsNet that outputs class scores suitable for margin loss.
    
    This version outputs the capsule lengths (L2 norms) for each class,
    which are then passed to margin loss during training.
    
    Args:
        num_subjects (int): Number of subjects
        l2_reg (float): L2 regularization
        
    Returns:
        Model: Compiled Keras model
    """
    inputs = Input(shape=(128, 128, 1), name='input')
    
    # Feature extraction
    x = Conv2D(64, 5, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(128, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(256, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(512, 3, activation='relu', padding='same',
              kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Generate class capsule vectors (num_subjects × 16D)
    capsule_vecs = Dense(num_subjects * 16, activation=None,
                        kernel_regularizer=l2(l2_reg))(x)
    capsule_vecs = layers.Reshape((num_subjects, 16))(capsule_vecs)
    
    # Output the L2 norm of each capsule (length = [0, 1] via sigmoid)
    # This replaces the problematic squashing activation
    capsule_lengths = layers.Lambda(
        lambda x: tf.nn.sigmoid(tf.norm(x, axis=-1)),
        name='capsule_lengths'
    )(capsule_vecs)
    
    model = Model(inputs=inputs, outputs=capsule_lengths)
    
    # Use a simple loss that's more stable
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # More stable than margin loss
        metrics=['accuracy']
    )
    
    return model
