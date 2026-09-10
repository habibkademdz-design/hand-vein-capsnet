"""Capsule Network Model for Hand Vein Recognition"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Reshape, Dense, Lambda, Flatten
from tensorflow.keras.regularizers import l2

from .layers import PrimaryCapsules, DynamicRouting
from .losses import MarginLoss


class CapsuleNetwork(Model):
    """Improved Capsule Network for Hand Vein Recognition.
    
    Args:
        num_subjects (int): Number of subjects (classes) (default: 276)
        dim_capsule (int): Dimension of class capsules (default: 16)
        routing_iterations (int): Number of routing iterations (default: 3)
        l2_reg (float): L2 regularization coefficient (default: 1e-4)
    """
    
    def __init__(self, 
                 num_subjects=276,
                 dim_capsule=16,
                 routing_iterations=3,
                 l2_reg=1e-4,
                 **kwargs):
        super(CapsuleNetwork, self).__init__(**kwargs)
        
        self.num_subjects = num_subjects
        self.dim_capsule = dim_capsule
        self.routing_iterations = routing_iterations
        self.l2_reg = l2_reg
        
        # Feature extraction layers
        self.conv1 = Conv2D(64, 5, activation='relu', padding='valid',
                           kernel_regularizer=l2(l2_reg))
        self.bn1 = BatchNormalization()
        
        self.conv2 = Conv2D(128, 3, activation='relu', padding='valid',
                           kernel_regularizer=l2(l2_reg))
        self.bn2 = BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        
        self.conv3 = Conv2D(256, 3, activation='relu', padding='valid',
                           kernel_regularizer=l2(l2_reg))
        self.bn3 = BatchNormalization()
        
        # Primary capsules
        self.primary_capsules = PrimaryCapsules(
            num_capsules=32,
            dim_capsules=8,
            kernel_size=9,
            strides=2
        )
        
        # Flatten layer
        self.flatten = Flatten()
        
        # Class capsules (dense connection)
        self.class_capsules = Dense(
            num_subjects * dim_capsule,
            activation=None,
            kernel_regularizer=l2(l2_reg)
        )
    
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs (tensor): Input images [batch_size, 128, 128, 1]
            training (bool): Training mode flag
            
        Returns:
            tuple: (capsule_lengths, class_capsules)
        """
        # Feature extraction
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        # Primary capsules
        u = self.primary_capsules(x, training=training)
        # u shape: [batch_size, num_primary_capsules, 8]
        
        # Flatten and pass through dense layer to get class capsules
        batch_size = tf.shape(u)[0]
        
        # Flatten the primary capsules
        u_flat = self.flatten(u)
        v_flat = self.class_capsules(u_flat)
        
        # Reshape back to capsules
        class_caps = tf.reshape(v_flat, [batch_size, self.num_subjects, self.dim_capsule])
        
        # Apply squashing
        class_caps = self.squash(class_caps)
        
        # Compute capsule lengths for output
        capsule_lengths = tf.norm(class_caps, axis=-1)
        
        return capsule_lengths, class_caps
    
    @staticmethod
    def squash(vectors, axis=-1):
        """Squashing activation function."""
        norm = tf.norm(vectors, axis=axis, keepdims=True)
        squashed = (1 - 1 / (1 + norm)) * vectors / (norm + 1e-7)
        return squashed
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_subjects': self.num_subjects,
            'dim_capsule': self.dim_capsule,
            'routing_iterations': self.routing_iterations,
            'l2_reg': self.l2_reg,
        })
        return config


def build_capsnet(num_subjects=276, l2_reg=1e-4):
    """Build complete CapsNet model.
    
    Args:
        num_subjects (int): Number of subjects
        l2_reg (float): L2 regularization
        
    Returns:
        Model: Compiled Keras model
    """
    inputs = Input(shape=(128, 128, 1))
    
    # Feature extraction
    x = Conv2D(64, 5, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = Conv2D(256, 3, activation='relu', padding='valid',
              kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    # Primary capsules
    primary_caps = PrimaryCapsules(
        num_capsules=32,
        dim_capsules=8,
        kernel_size=9,
        strides=2
    )(x)
    
    # Flatten using Flatten layer instead of Reshape(-1)
    flat = Flatten()(primary_caps)
    
    # Class capsules
    class_caps_flat = Dense(
        num_subjects * 16,
        activation=None,
        kernel_regularizer=l2(l2_reg)
    )(flat)
    
    class_caps = Reshape((num_subjects, 16))(class_caps_flat)
    
    # Apply squashing
    class_caps = Lambda(lambda x: CapsuleNetwork.squash(x))(class_caps)
    
    # Output: capsule lengths (for margin loss)
    capsule_lengths = Lambda(lambda x: tf.norm(x, axis=-1))(class_caps)
    
    model = Model(inputs=inputs, outputs=capsule_lengths)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5),
        metrics=['accuracy']
    )
    
    return model
