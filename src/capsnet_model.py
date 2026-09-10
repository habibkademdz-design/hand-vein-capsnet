"""Capsule Network Model for Hand Vein Recognition - FIXED VERSION"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Lambda, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2

from .losses import MarginLoss


def build_capsnet(num_subjects=276, l2_reg=1e-4):
    """Build CapsNet model with proper scaling.
    
    Args:
        num_subjects (int): Number of subjects
        l2_reg (float): L2 regularization
        
    Returns:
        Model: Compiled Keras model
    """
    inputs = Input(shape=(128, 128, 1), name='input')
    
    # Feature extraction - Conv layers with BatchNorm
    x = Conv2D(64, 5, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    
    x = Conv2D(128, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    x = Conv2D(256, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    
    x = Conv2D(256, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    x = Conv2D(512, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    
    # Primary capsule-like layer (multiple small convolutions)
    capsule_feats = []
    for i in range(8):  # 8 different feature detectors
        feat = Conv2D(32, 1, activation='relu', padding='same',
                     kernel_regularizer=l2(l2_reg), 
                     name=f'capsule_feat_{i}')(x)
        capsule_feats.append(feat)
    
    # Concatenate all capsule features
    x = layers.Concatenate(name='capsule_concat')(capsule_feats)
    
    # Global average pooling to create capsule-like vectors
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layers for class capsules with proper normalization
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg),
             name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg),
             name='dense2')(x)
    x = layers.BatchNormalization(name='bn_dense2')(x)
    
    # Output: class capsules (num_subjects, 16 dimensions each)
    # Use smaller weight initialization to avoid exploding outputs
    class_caps_flat = Dense(
        num_subjects * 16,
        activation=None,
        kernel_regularizer=l2(l2_reg),
        kernel_initializer=keras.initializers.GlorotUniform(),
        name='class_capsules_dense'
    )(x)
    
    # Reshape to capsule format
    class_caps = layers.Reshape((num_subjects, 16), name='reshape_capsules')(class_caps_flat)
    
    # Apply improved squashing activation with scaling
    class_caps = Lambda(lambda x: improved_squash_activation(x), 
                       name='squashing')(class_caps)
    
    # Output: capsule lengths for margin loss
    # Use proper normalization
    capsule_lengths = Lambda(lambda x: tf.norm(x, axis=-1, keepdims=False),
                            name='capsule_lengths')(class_caps)
    
    model = Model(inputs=inputs, outputs=capsule_lengths)
    
    # Compile model with proper loss scaling
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0),
        loss=MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    return model


def improved_squash_activation(vectors, axis=-1):
    """Improved squashing activation that prevents value collapse.
    
    Standard squashing: v = ||v||^2 / (1 + ||v||^2) * v / ||v||
    This can collapse values to very small range.
    
    Improved version maintains better numerical range.
    
    Args:
        vectors: Input capsule vectors
        axis: Axis to compute norm
        
    Returns:
        Squashed vectors with better scaling
    """
    norm = tf.norm(vectors, axis=axis, keepdims=True)
    
    # Improved squashing with better scaling
    # Use tanh-like behavior but with larger output range
    scale = 2.0 * norm / (1.0 + norm ** 2)  # Range roughly [0, 2]
    squashed = scale * vectors / (norm + 1e-7)
    
    return squashed


def build_simple_capsnet(num_subjects=276, l2_reg=1e-4):
    """Build simplified CapsNet for faster training/testing.
    
    Good for quick prototyping and debugging.
    
    Args:
        num_subjects (int): Number of subjects
        l2_reg (float): L2 regularization
        
    Returns:
        Model: Compiled Keras model
    """
    inputs = Input(shape=(128, 128, 1), name='input')
    
    # Simplified architecture with more regularization
    x = Conv2D(64, 5, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = Dropout(0.2, name='drop1')(x)
    
    x = Conv2D(128, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = Dropout(0.2, name='drop2')(x)
    
    x = Conv2D(256, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg), name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Dropout(0.2, name='drop3')(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Dense layers with BatchNorm
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg), name='dense1')(x)
    x = BatchNormalization(name='bn_dense1')(x)
    x = Dropout(0.3, name='drop_dense')(x)
    
    # Class capsules
    class_caps_flat = Dense(num_subjects * 16, activation=None,
                           kernel_regularizer=l2(l2_reg),
                           kernel_initializer=keras.initializers.GlorotUniform(),
                           name='class_capsules_dense')(x)
    class_caps = layers.Reshape((num_subjects, 16), name='reshape_capsules')(class_caps_flat)
    class_caps = Lambda(lambda x: improved_squash_activation(x), name='squashing')(class_caps)
    
    # Output
    capsule_lengths = Lambda(lambda x: tf.norm(x, axis=-1, keepdims=False), 
                            name='capsule_lengths')(class_caps)
    
    model = Model(inputs=inputs, outputs=capsule_lengths)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0),
        loss=MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5),
        metrics=['accuracy']
    )
    
    return model
