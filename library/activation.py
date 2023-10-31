import numpy as np
import tensorflow as tf

from library.error_handler import Error_Handler

class Activation:
    
    def forward_activation(self, act_type, layer):
        if act_type == "relu": 
            layer = Activation.relu(self, x=layer)
        elif act_type == "tanh":
            layer = Activation.tanh(self, x=layer)
        elif act_type == "softmax":
            layer = Activation.softmax(self, x=layer)
        elif act_type == "signum":
            layer = Activation.signum(self, x=layer)
        else:
            Error_Handler.unknown_forward_activation_function()
            return
        
        return layer
        
    
    def backward_activation(self, act_type, layer):
        if act_type == "relu": 
            layer = Activation.relu_derivative(self, x=layer)
            return layer
        elif act_type == "tanh":
            layer = Activation.tanh_derivative(self, x=layer)
            return layer
        elif act_type == "softmax":
            layer = Activation.softmax_derivative(self, x=layer)
            return layer
        else:
            Error_Handler.unknown_backward_activation_function()
            return
        
        return layer


    #Forward Activation Functions
    def relu(self, x):
        return tf.nn.relu(x)
    
    
    def tanh(self, x):
        return tf.nn.tanh(x)

    
    def signum(self, x):
        return tf.math.sign(x)

    
    def softmax(self, x):
        return tf.nn.softmax(x)


    #Backward Activation Functions
    def tanh_derivative(self, x):
        return (1 - (tf.nn.tanh(x) ** 2))
  
    
    def relu_derivative(self, x):
        return tf.where(x < 0.0, 0.0, 1.0)
