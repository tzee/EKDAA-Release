import numpy as np

from library.layer_types import Layer_Types
from library.FNN_formatter import FNN_Formatter
from library.error_handler import Error_Handler
from library.learning_mechs import Learning_Mechs
from library.FNN_builder_BP import FNN_Builder_BP
from library.FNN_builder_EKDAA import FNN_Builder_EKDAA

class FNN:
    
    model_outline                = []
    model_internal               = None
    internal_learning_rule       = None
    
    conv_layer_number            = 1
    max_pool_layer_number        = 1
    fully_connected_layer_number = 1 
    
    def __init__(self, model_name, gpu_number):
        self.model_name = model_name
        self.gpu_number = gpu_number

        
    def get_model(self):
        return self.model_internal

    
    def compute_tsne(self, train, labels, layer, bs, samples, label):
        return self.model_internal.compute_tsne(train, labels, layer, bs, samples, label)

    
    def randomize_weights(self):
        self.model_internal.randomize_weights()

        
    def add_input(self, input_shape):
        curr_input_layer = [Layer_Types.Input(), input_shape]
        self.model_outline.append(curr_input_layer)
        return
    
    
    def add_conv(self, filters, filter_size, activation, w_init, use_bias, b_init, dropout):
        curr_conv_layer = [Layer_Types.Conv(), self.conv_layer_number, filters, 
                           filter_size, activation, w_init, use_bias, b_init,
                           dropout]   
        
        self.model_outline.append(curr_conv_layer)
        self.conv_layer_number += 1
        return
    
    
    def add_maxpool(self, kernel, stride):
        curr_mp_layer = [Layer_Types.Maxpool(), self.max_pool_layer_number,
                         kernel, stride]
        self.model_outline.append(curr_mp_layer)
        self.max_pool_layer_number += 1
        return
    
    
    def flatten(self):
        curr_flatten = [Layer_Types.Flatten()]
        self.model_outline.append(curr_flatten)
        return
    
    
    def add_fc(self, nodes, activation, w_init, use_bias, b_init, dropout):
        curr_fc_layer = [Layer_Types.FC(), self.fully_connected_layer_number,
                         nodes, activation, w_init, use_bias, b_init, dropout]
        self.model_outline.append(curr_fc_layer)
        self.fully_connected_layer_number += 1
        return
    
    
    #Creation of the model
    def build(self, learning_mechanism):
        if learning_mechanism.upper() == Learning_Mechs.BP():
            self.model_internal = FNN_Builder_BP.build(self, self.model_outline)
            self.internal_learning_rule = Learning_Mechs.BP()
        elif learning_mechanism.upper() == Learning_Mechs.EKDAA():
            self.model_internal = FNN_Builder_EKDAA.build(self, self.model_outline)
            self.internal_learning_rule = Learning_Mechs.EKDAA()
        else:
            Error_Handler.invalid_learning_mechanism()
        return
    
    
    def set_optimizer(self, opt):
        if self.internal_learning_rule == Learning_Mechs.BP():
            FNN_Builder_BP.set_optimizer(self, opt)
        elif self.internal_learning_rule == Learning_Mechs.EKDAA():
            FNN_Builder_EKDAA.set_optimizer(self, opt)        
        else:
            Error_Handler.unknown_learning_mechanism_cannot_set_optimizer()
        return
    
    
    def set_beta(self, beta):
        if self.internal_learning_rule == Learning_Mechs.BP():
             Error_Handler.backprop_does_not_support_beta()
        elif self.internal_learning_rule == Learning_Mechs.EKDAA():
            FNN_Builder_EKDAA.set_beta(self, beta)        
        else:
            Error_Handler.learning_mech_does_not_support_beta()
        return
    
    
    def set_gamma(self, gamma):
        if self.internal_learning_rule == Learning_Mechs.BP():
             Error_Handler.backprop_does_not_support_gamma()
        elif self.internal_learning_rule == Learning_Mechs.EKDAA():
            FNN_Builder_EKDAA.set_gamma(self, gamma)        
        else:
            Error_Handler.learning_mech_does_not_support_gamma()
        return
        
    
    def forward(self, data, is_training):
        if self.internal_learning_rule == Learning_Mechs.BP():
            FNN_Builder_BP.forward(self, data, is_training)
        elif self.internal_learning_rule == Learning_Mechs.EKDAA():
            FNN_Builder_EKDAA.forward(self, data, is_training)        
        else:
            Error_Handler.unknown_learning_mechanism_cannot_compute_forward()
        return
  
        
    def backward(self, data, labels):
        if self.internal_learning_rule == Learning_Mechs.BP():
            FNN_Builder_BP.backward(self, data, labels)
        elif self.internal_learning_rule == Learning_Mechs.EKDAA():
            FNN_Builder_EKDAA.backward(self, data, labels)
        else:
            Error_Handler.unknown_learning_mechanism_cannot_compute_backward()
        return
            
        
    def update(self):
        if self.internal_learning_rule == Learning_Mechs.BP():
            FNN_Builder_BP.update(self)
        elif self.internal_learning_rule == Learning_Mechs.EKDAA():
            FNN_Builder_EKDAA.update(self)
        else:
            Error_Handler.unknown_learning_mechanism_cannot_compute_update()
        return
                
        
    def infer(self, data, labels, batch_size):
        if self.internal_learning_rule == Learning_Mechs.BP():
            return FNN_Builder_BP.infer(self, data, labels, batch_size)
        elif self.internal_learning_rule == Learning_Mechs.EKDAA():
            return FNN_Builder_EKDAA.infer(self, data, labels, batch_size)
        else:
            Error_Handler.unknown_learning_mechanism_cannot_compute_infer()
        return
       
        
    def print_model(self):
        print("------------- Model Configuration---------------")
        for i in range(np.shape(self.model_outline)[0]):
            if self.model_outline[i][0] == Layer_Types.Conv():
                print(FNN_Formatter.format_conv_layer(self.model_outline[i]))
            elif self.model_outline[i][0] == Layer_Types.Maxpool():
                print(FNN_Formatter.format_mp_layer(self.model_outline[i]))
            elif self.model_outline[i][0] == Layer_Types.FC():
                print(FNN_Formatter.format_fc_layer(self.model_outline[i]))
            elif self.model_outline[i][0] == Layer_Types.Input():
                print(FNN_Formatter.format_input_layer(self.model_outline[i]))   
        print("------------------------------------------------")
        return
    
