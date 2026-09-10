"""Dynamic Routing Algorithm"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DynamicRouting(layers.Layer):
    """Dynamic routing layer (routing-by-agreement).
    
    Args:
        num_routing_iterations (int): Number of routing iterations (default: 3)
        name (str): Layer name
    """
    
    def __init__(self, num_routing_iterations=3, **kwargs):
        super(DynamicRouting, self).__init__(**kwargs)
        self.num_routing_iterations = num_routing_iterations
    
    def build(self, input_shape):
        """
        Args:
            input_shape: tuple of (batch_size, num_lower_capsules, dim_lower_capsules, 
                                   num_upper_capsules, dim_upper_capsules, weight_matrix)
        """
        assert len(input_shape) == 2
        self.num_lower_capsules = input_shape[0][1]
        self.num_upper_capsules = input_shape[1][1]
        self.dim_lower_capsules = input_shape[0][-1]
        self.dim_upper_capsules = input_shape[1][-1]
        
        super(DynamicRouting, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: [u_predict, W] where:
                - u_predict: predicted vectors [batch, num_lower, num_upper, dim_upper]
                - W: weight matrix [num_lower, num_upper, dim_lower, dim_upper]
        
        Returns:
            v: class capsule vectors [batch, num_upper, dim_upper]
        """
        u_predict, W = inputs
        batch_size = tf.shape(u_predict)[0]
        
        # Initialize routing coefficients
        b = tf.zeros([batch_size, self.num_lower_capsules, self.num_upper_capsules])
        
        # Routing iterations
        for iteration in range(self.num_routing_iterations):
            # Compute routing weights (softmax)
            c = tf.nn.softmax(b, axis=-1)  # [batch, num_lower, num_upper]
            
            # Compute weighted sum of predictions
            # c: [batch, num_lower, num_upper, 1]
            # u_predict: [batch, num_lower, num_upper, dim_upper]
            c_expanded = tf.expand_dims(c, axis=-1)
            s = tf.reduce_sum(c_expanded * u_predict, axis=1)  # [batch, num_upper, dim_upper]
            
            # Apply squashing
            v = self.squash(s)  # [batch, num_upper, dim_upper]
            
            # Update routing coefficients (via agreement)
            if iteration < self.num_routing_iterations - 1:
                # Compute agreement (dot product)
                # v_expanded: [batch, 1, num_upper, dim_upper]
                v_expanded = tf.expand_dims(v, axis=1)
                
                # agreement: [batch, num_lower, num_upper]
                agreement = tf.reduce_sum(u_predict * v_expanded, axis=-1)
                b = b + agreement
        
        return v
    
    @staticmethod
    def squash(vectors, axis=-1):
        """Squashing activation function.
        
        Args:
            vectors (tensor): Input vectors
            axis (int): Axis to compute norm
            
        Returns:
            tensor: Squashed vectors
        """
        norm = tf.norm(vectors, axis=axis, keepdims=True)
        squashed = (1 - 1 / (1 + norm)) * vectors / (norm + 1e-7)
        return squashed
    
    def get_config(self):
        config = super(DynamicRouting, self).get_config()
        config.update({
            'num_routing_iterations': self.num_routing_iterations,
        })
        return config
