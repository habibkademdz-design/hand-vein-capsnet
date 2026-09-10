"""Primary Capsule Layer"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PrimaryCapsules(layers.Layer):
    """Primary Capsule layer that reshapes and applies squashing.
    
    Args:
        num_capsules (int): Number of capsule types (default: 32)
        dim_capsules (int): Dimension of each capsule (default: 8)
        kernel_size (int): Convolutional kernel size (default: 9)
        strides (int): Convolutional stride (default: 2)
        activation (str): Activation function (default: 'relu')
    """
    
    def __init__(self, 
                 num_capsules=32,
                 dim_capsules=8,
                 kernel_size=9,
                 strides=2,
                 activation='relu',
                 **kwargs):
        super(PrimaryCapsules, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        
        # Convolution layer outputs num_capsules * dim_capsules channels
        self.conv2d = layers.Conv2D(
            filters=num_capsules * dim_capsules,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            activation=activation
        )
    
    def build(self, input_shape):
        self.conv2d.build(input_shape)
        super(PrimaryCapsules, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """Forward pass through primary capsule layer.
        
        Args:
            inputs (tensor): Input tensor [batch_size, height, width, channels]
            training (bool): Training mode flag
            
        Returns:
            tensor: Primary capsules [batch_size, num_capsules, height, width, dim_capsules]
                   reshaped to [batch_size, num_capsules * height * width, dim_capsules]
        """
        # Apply convolution
        output = self.conv2d(inputs, training=training)
        
        # Get shape
        batch_size = tf.shape(output)[0]
        height = tf.shape(output)[1]
        width = tf.shape(output)[2]
        
        # Reshape: [batch, height, width, num_capsules * dim_capsules]
        # -> [batch, height, width, num_capsules, dim_capsules]
        output = tf.reshape(output, 
                           [batch_size, height, width, self.num_capsules, self.dim_capsules])
        
        # Transpose to [batch, num_capsules, height, width, dim_capsules]
        output = tf.transpose(output, [0, 3, 1, 2, 4])
        
        # Reshape to [batch, num_capsules * height * width, dim_capsules]
        output = tf.reshape(output, [batch_size, -1, self.dim_capsules])
        
        # Apply squashing
        output = self.squash(output)
        
        return output
    
    @staticmethod
    def squash(vectors, axis=-1):
        """Squashing activation function.
        
        Args:
            vectors (tensor): Input vectors
            axis (int): Axis to compute norm
            
        Returns:
            tensor: Squashed vectors
        """
        # Compute norm
        norm = tf.norm(vectors, axis=axis, keepdims=True)
        
        # Squashing: v = ||v||^2 / (1 + ||v||^2) * v / ||v||
        squashed = (1 - 1 / (1 + norm)) * vectors / (norm + 1e-7)
        
        return squashed
    
    def get_config(self):
        config = super(PrimaryCapsules, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsules': self.dim_capsules,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.activation,
        })
        return config
