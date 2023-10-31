import tensorflow as tf

from library.error_handler import Error_Handler

class Weight_Initialization:
    
    def build_weights(self, weight_init, shape):
        weights = []
        if weight_init == "glorot_uniform": 
            weights = Weight_Initialization.glorot_uniform_initialization(self, w_shape=shape)
        elif weight_init == "glorot_normal":
            weights = Weight_Initialization.glorot_normal_initialization(self, w_shape=shape)
        else:
            Error_Handler.weight_initialization_type_not_understood()
        return weights


    def glorot_uniform_initialization(self, w_shape):
        weights = tf.Variable(tf.initializers.GlorotUniform()(shape=w_shape, dtype=tf.float32)) 
        return weights
     
        
    def glorot_normal_initialization(self, w_shape):
        weights = tf.Variable(tf.initializers.GlorotNormal()(shape=w_shape, dtype=tf.float32))
        return weights
