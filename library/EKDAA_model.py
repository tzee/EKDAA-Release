import numpy as np 
import tensorflow as tf

from library.Z import Z
from library.core import Core
from library.validator import Validator
from library.layer_types import Layer_Types
from library.normalization import Normalization
from library.error_handler import Error_Handler

core = Core()
normalization = Normalization()

class EKDAA_Model:
    
    optimizer   = None
    has_flatten = False
    has_conv    = False
  
    post_forward_flat = False

    #Default values
    BETA        = 0.9
    GAMMA       = 0.1
    
    #Forward pass 
    h           = []
    z           = []
    weights     = []
    use_b       = []
    biases      = []
    activations = []
    dropout     = []
    l_type      = []
    
    #Backward pass 
    loss_soft = []
    e         = []
    E         = []
    y         = []
    del_w     = []
    del_b     = []
    del_E     = []
    passback  = []
    
    def randomize_weights(self):
        for i in range(len(self.weights)):
            self.weights[i].assign(tf.Variable(core.set_weights(weights=Z.glorot_uniform(), shape=np.shape(self.weights[i]))))
            
    def add_type(self, l_type):
        self.l_type.append(l_type)

    
    def add_layer(self, layer):
        self.h.append(tf.Variable(layer))
        self.z.append(tf.Variable(layer))
        self.e.append(tf.Variable(layer*0))
        self.y.append(tf.Variable(layer*0))

        
    def add_weights(self, batch_size, weights, before_pool=False):
        self.weights.append(tf.Variable(weights))
        self.del_w.append(tf.Variable(weights*0))
        self.E.append(tf.Variable(tf.transpose(weights)))
        self.del_E.append(tf.Variable(tf.transpose(weights*0)))
        
        if weights.numpy().ndim > 2:
            fil_size, fil_size2, in_fil, out_fil = np.shape(weights)
            batch_size, width, height, channels = np.shape(self.h[-1])
            
            if self.l_type[-1] == Layer_Types.Maxpool():
                if before_pool:
                    self.passback.append(tf.Variable(tf.zeros([batch_size, width*2, height*2, in_fil])))
                else:
                    self.passback.append(tf.Variable(tf.zeros([batch_size, width, height, in_fil])))
            else:
                self.passback.append(tf.Variable(tf.zeros([batch_size, width, height, in_fil])))
        else:
            self.passback.append(tf.Variable(tf.zeros([batch_size, np.shape(weights)[0]])))      
        
        
    def use_biases(self, use_biases):
        self.use_b.append(use_biases)
        
        
    def add_biases(self, biases):
        self.biases.append(tf.Variable(biases))
        self.del_b.append(tf.Variable(biases*0))
        
        
    def add_activations(self, act_type):
        self.activations.append(act_type)
    
    
    def add_dropout(self, dropout_rate):
        self.dropout.append(dropout_rate)


    def get_model(self):
        return self.z
    
    
    def get_weights(self):
        return self.weights
    
    
    def set_optimizer(self, opt):
        self.optimizer = opt
        
        
    def set_beta(self, beta):
        self.BETA = beta
    
    
    def set_gamma(self, gamma):
        self.GAMMA = gamma
    
    
    def forward(self, data, is_training):
       try:
            layers_forward = len(self.z)
            
            #2 layers needed for flattening
            if self.has_flatten:
                layers_forward -= 1
            
            for i in range(layers_forward):
                if i == 0:
                    self.h[0].assign(data)
                    self.z[0].assign(data)  
                elif self.has_conv and not self.post_forward_flat and not Layer_Types.Flatten() in self.l_type[i]:
                    #Maxpool
                    if self.l_type[i] == Layer_Types.Maxpool():
                        self.h[i].assign(core.maxpool(self.h[i-1]))
                        self.z[i].assign(self.h[i])
                    #Convolution
                    else:
                        self.h[i].assign(core.conv(self.h[i-1], self.weights[i-1]))            
                        if self.use_b[i-1]:
                            self.h[i].assign(tf.add(self.h[i], self.biases[i-1]))
                            
                        self.z[i].assign(core.activate(self.activations[i-1], self.h[i]))
                        
                elif Layer_Types.Flatten() in self.l_type[i]: 
                    self.h[i+1].assign(tf.reshape(self.h[i-1], shape=(np.shape(self.h[i-1])[0], -1)))
                    self.z[i+1].assign(self.h[i+1])
                    self.post_forward_flat = True 
                else:
                    if self.has_flatten and self.post_forward_flat: 
                        self.h[i+1].assign(tf.matmul(self.z[i], self.weights[i-2]))
                        if self.use_b[i-2]:
                            self.h[i+1].assign(tf.add(self.h[i+1], self.biases[i-2]))
                            
                        self.z[i+1].assign(core.activate(self.activations[i-2], self.h[i+1]))
                                                
                    else:
                        self.h[i].assign(tf.matmul(self.z[i-1], self.weights[i-1]))
                        if self.use_b[i-1]:
                            self.h[i].assign(tf.add(self.h[i], self.biases[i-1]))
                        
                        self.z[i].assign(core.activate(self.activations[i-1], self.h[i]))  
          
            self.post_forward_flat = False
       except:
            Error_Handler.unknown_error_in_forward()
    
        
    def backward(self, data, labels):    
         try:       
             #Compute Targets
             init_ekdaa = True
             layers_backward = len(self.z)-2
             
             backwards_stop = -1 
            
             if self.has_flatten:
                 backwards_stop += 1    

             #-2 account for input layer 
             for i in range(layers_backward, backwards_stop, -1): 
                 if init_ekdaa:                       
                     if labels.ndim != 2:
                         loss_and_softmax_dir = (self.z[i+1] - labels[:,0])
                         labels = labels[:,0]
                     else:
                         loss_and_softmax_dir = (self.z[i+1] - labels)
               
                     if self.has_flatten:
                         self.y[i+1].assign(labels) 
                         self.e[i+1].assign(loss_and_softmax_dir)

                         self.passback[i-2].assign(tf.matmul(self.e[i+1], self.E[i-2]))
                         self.passback[i-2].assign(self.passback[i-2] * self.BETA)
                         self.passback[i-2].assign(tf.subtract(self.h[i], self.passback[i-2]))
                     else:
                         self.y[i+1].assign(labels) 
                         self.e[i+1].assign(tf.subtract(self.z[i+1], self.y[i+1]))
                                      
                         self.passback[i].assign(tf.matmul(self.e[i+1], self.E[i]))
                         self.passback[i].assign(self.passback[i] * self.BETA)
                         self.passback[i].assign(tf.subtract(self.h[i], self.passback[i]))
                                                              
                     init_ekdaa = False
                 else:
                     if self.has_flatten:       
                         if self.l_type[i] == Layer_Types.FC():
                             self.y[i+1].assign(core.activate(self.activations[0], self.passback[i-1]))
                             self.e[i+1].assign(tf.subtract(self.z[i+1], self.y[i+1]))
                           
                             self.passback[i-2].assign(tf.matmul(self.e[i+1], self.E[i-2]))
                             self.passback[i-2].assign(self.passback[i-2] * self.BETA)
                             self.passback[i-2].assign(tf.subtract(self.h[i], self.passback[i-2]))
                        
                         #Maxpool
                         elif self.l_type[i] == Layer_Types.Maxpool():
                             self.passback[i-1].assign(core.pool_upsample(tf.reshape(self.passback[i], shape=(np.shape(self.z[i]))), 2, 2, "SAME"))
                         #Convolution
                         elif self.l_type[i] == Layer_Types.Conv():       
                             self.y[i].assign(core.activate(self.activations[0], tf.reshape(self.passback[i], 
                                                                                            shape=np.shape(self.z[i]))))
                             self.e[i].assign(tf.subtract(self.z[i], self.y[i]))

                             self.passback[i-1].assign(core.deconv_pb(self.passback[i-1], self.e[i], tf.transpose(self.E[i-1], perm=[2,3,1,0])))
                             self.passback[i-1].assign(self.passback[i-1] * self.BETA)
                             self.passback[i-1].assign(tf.subtract(self.h[i-1], self.passback[i-1]))
                                                                                 
                     else:
                         self.y[i+1] = core.activate(self.activations[0], self.passback[i+1])
                         self.e[i+1].assign(tf.subtract(self.z[i+1], self.y[i+1]))
                    
                         self.passback[i].assign(tf.matmul(self.e[i+1], self.E[i]))
                         self.passback[i].assign(self.passback[i] * self.BETA)
                         self.passback[i].assign(tf.subtract(self.h[i], self.passback[i]))
                                                            
             for i in range(len(self.del_w)-1, -1, -1): 
                 if self.has_flatten:
                     if self.del_w[i].numpy().ndim == 4:
                         #pooling [X, 1, X, 1]
                         if np.shape(self.del_w[i])[3] != 1:               
                             self.del_w[i].assign(core.deconv(self.del_w, i, self.z[i], self.e[i+1],
                                                              np.shape(self.weights[i])[2], np.shape(self.weights[i])[0]))        
                             self.del_b[i].assign(self.e[i+1] / np.max(self.e[i+1]))
                             self.del_E[i].assign(self.GAMMA * tf.transpose(self.del_w[i]))
                     else:
                         if i != 0:
                             self.del_w[i].assign(tf.matmul(tf.transpose(self.z[i+2]), self.e[i+3])) 
                             self.del_b[i].assign(self.e[i+3] / np.max(self.e[i+3]))
                             self.del_E[i].assign(self.GAMMA * tf.transpose(self.del_w[i]))
                         else:
                             self.del_w[i].assign(tf.matmul(tf.transpose(tf.cast(self.h[2], tf.float32)), self.e[i+3]))
                             self.del_b[i].assign(self.e[i+3] / np.max(self.e[i+3]))
                             self.del_E[i].assign(self.GAMMA * tf.transpose(self.del_w[i]))
                 else:
                     for i in range(len(self.z)-2, -1, -1): 
                         self.del_w[i].assign(tf.matmul(tf.transpose(self.z[i]), self.e[i+1]))
    
                     for i in range(len(self.z)-3, -1, -1): 
                         self.del_E[i].assign(self.GAMMA * tf.transpose(self.del_w[i]))       
         except:
            Error_Handler.unknown_error_in_backward()


    def apply_dropout(tensor, dropout_rate):
        dropped_indxs = tf.keras.backend.random_binomial(shape=np.shape(tensor), p=1-dropout_rate, dtype=tf.float32)
        tensor.assign(tensor*dropped_indxs)
        return tensor
    
    
    def update(self):
        if self.optimizer is not None:
            self.optimizer.apply_gradients(zip(self.del_w, self.weights))
            self.optimizer.apply_gradients(zip(self.del_b, self.biases))
            self.optimizer.apply_gradients(zip(self.del_E, self.E))
        else:
            Error_Handler.model_cannot_be_updated_optimizier_not_set()


    def compute_tsne(self, data, labels, output_layer, batch_size, total_samples, tsne_save_file_path):
        try:
            raw_data = np.shape(self.h[output_layer][0,])

            batches = int(total_samples/batch_size)
            tsne_outputs = 0
            
            if len(raw_data) == 3:
                tsne_outputs = np.zeros(shape=(total_samples, raw_data[0], raw_data[1], raw_data[2]))
            elif len(raw_data) == 1:
                tsne_outputs = np.zeros(shape=(total_samples, raw_data[0]))

            for i in range(batches):
                EKDAA_Model.forward(self, data[i*batch_size:(i+1)*batch_size], is_training=False)
                predictions = self.z[output_layer]
                tsne_outputs[i*batch_size:(i+1)*batch_size] = predictions.numpy()

            return tsne_outputs, labels
        except:
            print('Error when computing t-SNE for EKDAA model.')

            
            
    def infer(self, data, labels, batch_size):
        samples = np.shape(data)[0]
        batches = int(samples/batch_size)
        
        if not Validator.validate_batch_size(samples, batches):
            Error_Handler.invalid_partial_batch()
        
        total_accuracy = 0
        
        for i in range(batches):
            EKDAA_Model.forward(self, data[i*batch_size:(i+1)*batch_size], is_training=False)
            predictions = self.z[-1]
            predictions = tf.argmax(predictions, axis=1)
                        
            accuracy = core.compute_accuracy(predictions, labels[i*batch_size:(i+1)*batch_size])
            total_accuracy += accuracy
        accuracy_percentage = ((total_accuracy/batches) * 100)
        return accuracy_percentage
    
    

    
