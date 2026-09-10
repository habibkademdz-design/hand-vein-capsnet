"""Custom loss functions for CapsNet"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class MarginLoss(keras.losses.Loss):
    """Margin loss for class capsules.
    
    Args:
        m_plus (float): Margin for positive class (default: 0.9)
        m_minus (float): Margin for negative class (default: 0.1)
        lambda_ (float): Weight for negative loss (default: 0.5)
    """
    
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5, **kwargs):
        super(MarginLoss, self).__init__(**kwargs)
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
    
    def call(self, y_true, y_pred):
        """Calculate margin loss.
        
        Args:
            y_true (tensor): One-hot encoded true labels [batch_size, num_classes]
            y_pred (tensor): Predicted capsule lengths [batch_size, num_classes]
            
        Returns:
            tensor: Scalar loss value
        """
        # y_pred shape: (batch_size, num_classes)
        # y_true shape: (batch_size, num_classes)
        
        # Positive loss: (m_plus - ||v||)^2 for presence of class
        positive_loss = y_true * tf.square(tf.maximum(0., self.m_plus - y_pred))
        
        # Negative loss: (||v|| - m_minus)^2 for absence of class
        negative_loss = (1 - y_true) * tf.square(tf.maximum(0., y_pred - self.m_minus))
        
        # Combined loss
        loss = positive_loss + self.lambda_ * negative_loss
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'm_plus': self.m_plus,
            'm_minus': self.m_minus,
            'lambda_': self.lambda_,
        })
        return config


class CapsuleNetLoss(keras.losses.Loss):
    """Combined loss function for CapsNet with L2 regularization.
    
    Args:
        margin_loss_weight (float): Weight for margin loss (default: 1.0)
        l2_weight (float): Weight for L2 regularization (default: 0.0)
    """
    
    def __init__(self, margin_loss_weight=1.0, l2_weight=0.0, **kwargs):
        super(CapsuleNetLoss, self).__init__(**kwargs)
        self.margin_loss_weight = margin_loss_weight
        self.l2_weight = l2_weight
        self.margin_loss = MarginLoss()
    
    def call(self, y_true, y_pred):
        """Calculate combined loss.
        
        Args:
            y_true (tensor): One-hot encoded true labels
            y_pred (tensor): Predicted capsule lengths
            
        Returns:
            tensor: Scalar loss value
        """
        margin_loss = self.margin_loss(y_true, y_pred)
        total_loss = self.margin_loss_weight * margin_loss
        
        return total_loss


class RoutingByAgreement(keras.losses.Loss):
    """Routing-by-agreement entropy regularization."""
    
    def __init__(self, weight=0.005, **kwargs):
        super(RoutingByAgreement, self).__init__(**kwargs)
        self.weight = weight
    
    def call(self, routing_coefficients):
        """Calculate entropy regularization.
        
        Args:
            routing_coefficients (tensor): Routing softmax coefficients
            
        Returns:
            tensor: Scalar regularization loss
        """
        entropy = -tf.reduce_sum(
            routing_coefficients * tf.math.log(routing_coefficients + 1e-10),
            axis=-1
        )
        return self.weight * tf.reduce_mean(entropy)
