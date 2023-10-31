import numpy as np 
import tensorflow as tf

from library.core import Core
from library.validator import Validator
from library.EKDAA_model import EKDAA_Model
from library.layer_types import Layer_Types
from library.error_handler import Error_Handler

core = Core() 
validator = Validator()

class FNN_Builder_EKDAA:
    
    model_is_built = False
    internal_model = EKDAA_Model()
    
    def build(self, model_outline):
        FNN_Builder_EKDAA.build_forward(self, model_outline)
        return FNN_Builder_EKDAA.internal_model

    
    def build_forward(self, model_outline):
        for i in range(len(model_outline)):
            if model_outline[i][0] == Layer_Types.Conv():
                FNN_Builder_EKDAA.add_conv(model_outline[i])
            elif model_outline[i][0] == Layer_Types.Maxpool():
                FNN_Builder_EKDAA.add_maxpool(model_outline[i])
            elif model_outline[i][0] == Layer_Types.Flatten():
                FNN_Builder_EKDAA.flatten(model_outline[i])
            elif model_outline[i][0] == Layer_Types.FC():
                FNN_Builder_EKDAA.add_fc(model_outline[i])
            elif model_outline[i][0] == Layer_Types.Input():
                FNN_Builder_EKDAA.add_init(self, model_outline[i])
                
        if validator.valid_model_structure(model_outline):
            FNN_Builder_EKDAA.model_is_built = True
        else:
            Error_Handler.invalid_model_structure_to_build()
        return
    
    
    def add_conv(layer):
        new_filters = layer[2]
        f_size      = layer[3] 
        act_type    = layer[4]
        w_init_type = layer[5]     
        use_b       = layer[6]
        b_init_type = layer[7]
        dropout     = layer[8]

        
        batch_size, width, height, cur_channels = np.shape(FNN_Builder_EKDAA.internal_model.get_model()[-1])

        new_layer = tf.Variable(tf.zeros(shape=(batch_size, width, height, new_filters)))
        new_weights = tf.Variable(core.set_weights(weight_init=w_init_type, shape=(f_size, f_size, cur_channels, new_filters)))
             
        FNN_Builder_EKDAA.internal_model.has_conv = True
                    
        FNN_Builder_EKDAA.internal_model.add_type(Layer_Types.Conv())
        FNN_Builder_EKDAA.internal_model.add_layer(new_layer)
        FNN_Builder_EKDAA.internal_model.add_weights(batch_size, new_weights)
        FNN_Builder_EKDAA.internal_model.add_activations(act_type)
        FNN_Builder_EKDAA.internal_model.use_biases(use_b)
        FNN_Builder_EKDAA.internal_model.add_dropout(dropout)
        
        new_biases = tf.Variable(core.set_biases(bias_init=b_init_type, shape=(batch_size, width, height, new_filters)))
        FNN_Builder_EKDAA.internal_model.add_biases(new_biases)   
        return
    
    
    def add_maxpool(layer):
        kernel = layer[2]
        stride = layer[3] 
        
        if (kernel == 2) and (stride == 2):
            batch_size, width, height, cur_channels = np.shape(FNN_Builder_EKDAA.internal_model.get_model()[-1])
    
            try: 
                new_width = int((width - kernel) / stride + 1)
                new_height = int((height - kernel) / stride + 1)
                
                #Not used in actual computation graph
                act_type = 'maxpool'
                use_b    = True
                b_init_type = 'glorot_uniform'
                
                new_layer = tf.Variable(tf.zeros(shape=(batch_size, new_width, new_height, cur_channels)))
                new_weights = tf.Variable(tf.zeros(shape=(batch_size, 1, cur_channels, 1)))
                new_biases = tf.Variable(core.set_biases(bias_init=b_init_type, shape=(1, 1)))
      
                FNN_Builder_EKDAA.internal_model.add_type(Layer_Types.Maxpool())
                FNN_Builder_EKDAA.internal_model.add_layer(new_layer)
                
                FNN_Builder_EKDAA.internal_model.add_weights(batch_size, new_weights, True)
        
                FNN_Builder_EKDAA.internal_model.add_activations(act_type)
                FNN_Builder_EKDAA.internal_model.use_biases(use_b)
                FNN_Builder_EKDAA.internal_model.add_dropout(-1)
          
                FNN_Builder_EKDAA.internal_model.add_biases(new_biases)   
                        
            except:
                Error_Handler.maxpool_can_only_be_completed_on_convolutional_layer()
        else:
            Error_Handler.maxpool_parameters_cannot_be_custom()
        return
    
    
    def flatten(layer):
        FNN_Builder_EKDAA.internal_model.add_layer(-1)
        
        four_d = np.shape(FNN_Builder_EKDAA.internal_model.get_model()[-2])
        flat_shape = tf.Variable(tf.zeros(shape=(four_d[0], four_d[1]*four_d[2]*four_d[3])))

        FNN_Builder_EKDAA.internal_model.add_type(Layer_Types.Flatten())
        FNN_Builder_EKDAA.internal_model.add_layer(flat_shape)
        FNN_Builder_EKDAA.internal_model.add_dropout(-1)
        FNN_Builder_EKDAA.internal_model.add_dropout(-1)
    
        FNN_Builder_EKDAA.internal_model.has_flatten = True    
        return 
    
    
    def add_fc(layer):    
        if Layer_Types.Flatten() in str(FNN_Builder_EKDAA.internal_model.get_model()[-1].numpy()):
            batch_size, width, height, channels = np.shape(FNN_Builder_EKDAA.internal_model.get_model()[-2])
            last_nodes = (width * height * channels)
        else:
            batch_size, last_nodes = np.shape(FNN_Builder_EKDAA.internal_model.get_model()[-1])

        next_nodes  = layer[2]
        act_type    = layer[3]
        w_init_type = layer[4]
        use_b       = layer[5]
        b_init_type = layer[6]
        dropout     = layer[7]

        new_layer = tf.Variable(tf.zeros(shape=(batch_size, next_nodes)))
        new_weights = tf.Variable(core.set_weights(weight_init=w_init_type, shape=(last_nodes, next_nodes)))
             
        FNN_Builder_EKDAA.internal_model.add_type(Layer_Types.FC())
        FNN_Builder_EKDAA.internal_model.add_layer(new_layer)
        FNN_Builder_EKDAA.internal_model.add_weights(batch_size, new_weights)
        
        FNN_Builder_EKDAA.internal_model.add_activations(act_type)
        FNN_Builder_EKDAA.internal_model.use_biases(use_b)
        FNN_Builder_EKDAA.internal_model.add_dropout(dropout)

        new_biases = tf.Variable(core.set_biases(bias_init=b_init_type, shape=(batch_size, next_nodes)))
        FNN_Builder_EKDAA.internal_model.add_biases(new_biases)   
        return
    
    
    def add_init(self, layer):
        input_shape = layer[1]
        init_layer = tf.Variable(tf.zeros(shape=(input_shape)), dtype=tf.float32)

        FNN_Builder_EKDAA.internal_model.add_type(Layer_Types.Input())
        FNN_Builder_EKDAA.internal_model.add_layer(init_layer)
        FNN_Builder_EKDAA.internal_model.add_dropout(-1)
        return
    
    
    def set_optimizer(self, opt):
        FNN_Builder_EKDAA.internal_model.set_optimizer(opt)
    
    
    def set_beta(self, beta):
        FNN_Builder_EKDAA().internal_model.set_beta(beta)
        
        
    def set_gamma(self, gamma):
        FNN_Builder_EKDAA.internal_model.set_gamma(gamma)
  
        
    def forward(self, data, is_training):
        if FNN_Builder_EKDAA.model_is_built:
            FNN_Builder_EKDAA.internal_model.forward(data, is_training)
        else:
            Error_Handler.invalid_model_execution_model_not_built()
          
    
    def backward(self, data, labels):
        if FNN_Builder_EKDAA.model_is_built:
            FNN_Builder_EKDAA.internal_model.backward(data, labels)
        else:
            Error_Handler.invalid_model_execution_model_not_built()
            
            
    def update(self):
        FNN_Builder_EKDAA.internal_model.update()
        
        
    def infer(self, data, labels, batch_size):
        return FNN_Builder_EKDAA.internal_model.infer(data, labels, batch_size)
