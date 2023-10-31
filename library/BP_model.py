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

class BP_Model:
    
    optimizer   = None
    has_flatten = False
    has_conv    = False
    
    post_forward_flat = False
    
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
    del_w     = []
    del_b     = []
    deconv    = []
    passback  = []
    
    def randomize_weights(self):
        for i in range(len(self.weights)):
            self.weights[i].assign(tf.Variable(core.set_weights(weight_init=Z.glorot_uniform(), shape=np.shape(self.weights[i]))))

            
    def add_type(self, l_type):
        self.l_type.append(l_type)

    
    def add_layer(self, layer):
        self.h.append(tf.Variable(layer))
        self.z.append(tf.Variable(layer))

        
    def add_weights(self, batch_size, weights, before_pool=False):
        self.weights.append(tf.Variable(weights))
        self.del_w.append(tf.Variable(weights*0))
        
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
        return self.h
    
    
    def get_weights(self):
        return self.weights
    
    
    def set_optimizer(self, opt):
        self.optimizer = opt
    
    
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
                            self.h[i].assign(tf.add(self.h[i], np.sum(self.biases[i-1])))
                            
                        self.z[i].assign(core.activate(self.activations[i-1], self.h[i]))
                        
                elif Layer_Types.Flatten() in self.l_type[i]:
                    self.h[i+1].assign(tf.reshape(self.h[i-1], shape=(np.shape(self.h[i-1])[0], -1)))
                    self.z[i+1].assign(self.h[i+1])
                    self.post_forward_flat = True 
                else:
                    if self.has_flatten and self.post_forward_flat: 
                        self.h[i+1].assign(tf.matmul(self.z[i], self.weights[i-2]))
                        if self.use_b[i-2]:
                            self.h[i+1].assign(tf.add(self.h[i+1], np.sum(self.biases[i-2])))
                            
                        self.z[i+1].assign(core.activate(self.activations[i-2], self.h[i+1]))
                                                
                    else:
                        if self.dropout[i-1] > 0 and is_training:
                            self.weights[i-1].assign(BP_Model.apply_dropout(self.weights[i-1], self.dropout[i]))
 
                        self.h[i].assign(tf.matmul(self.z[i-1], self.weights[i-1]))
                        if self.use_b[i-1]:
                            self.h[i].assign(tf.add(self.h[i], np.sum(self.biases[i-1])))
                        
                        self.z[i].assign(core.activate(self.activations[i-1], self.h[i]))  
          
            self.post_forward_flat = False
        except:
             Error_Handler.unknown_error_in_forward()
            
            
    def backward(self, data, labels):    
          try:              
            init_bp = True
            layers_backward = len(self.z)-2
            
            backwards_stop = -1 
            
            if self.has_flatten:
                backwards_stop += 1         
            
            #-2 account for input layer 
            for i in range(layers_backward, backwards_stop, -1): 
                if init_bp:
                    if labels.ndim != 2:
                        loss_and_softmax_dir = (self.z[i+1] - labels[:,0])
                    else:
                        loss_and_softmax_dir = (self.z[i+1] - labels)

                    if self.has_flatten:                              
                        self.del_w[i-2].assign(tf.matmul(tf.transpose(self.z[i]), loss_and_softmax_dir))
                        if self.use_b[i-2]:
                            self.del_b[i-2].assign(np.average(loss_and_softmax_dir, axis=1))
                            
                        self.passback[i-2].assign(tf.matmul(loss_and_softmax_dir, tf.transpose(self.weights[i-2])))
                        self.passback[i-2].assign(self.passback[i-2] * core.activate_backward(self.activations[0], self.z[i]))
                    else:
                        self.del_w[i].assign(tf.matmul(tf.transpose(self.z[i]), loss_and_softmax_dir))
                        if self.use_b[i]:
                            self.del_b[i].assign(loss_and_softmax_dir)
                    
                        self.passback[i].assign(tf.matmul(loss_and_softmax_dir, tf.transpose(self.weights[i])))
                        self.passback[i].assign(self.passback[i] * core.activate_backward(self.activations[0], self.z[i]))
    
                    init_bp = False
    
                else:
                    if self.has_flatten:       
                        if not (Layer_Types.Flatten() in self.l_type[i]):
                            if self.z[i].numpy().ndim != 2:     
                                #Maxpool
                                if self.l_type[i] == Layer_Types.Maxpool():   
                                    self.passback[i-1].assign(core.pool_upsample(tf.reshape(self.passback[i], shape=(np.shape(self.z[i]))), 2, 2, "SAME"))
                                #Convolution
                                else:
                                    self.del_w[i-1].assign(core.deconv(self.del_w, i-1, tf.reshape(self.passback[i-1], shape=np.shape(self.z[i-1])), self.z[i], np.shape(self.weights[i-1])[2], np.shape(self.weights[i-1])[0]))
                                    if self.use_b[i-1]:                                               
                                        self.del_b[i-1].assign(tf.reshape(self.passback[i-1], shape=np.shape(self.z[i-1])))
                                    
                                    if self.passback[i-1].numpy().ndim != 2:
                                        self.passback[i-1].assign(core.deconv_pb(self.passback[i-1], tf.reshape(self.passback[i], shape=np.shape(self.z[i])), self.weights[i-1]))
                                        self.passback[i-1].assign(self.passback[i-1] * core.activate_backward(self.activations[0], self.z[i-1]))
                            else:
                                self.del_w[i-2].assign(tf.matmul(tf.transpose(self.z[i]), self.passback[i-1]))
                                if self.use_b[i-2]:
                                    self.del_b[i-2].assign(np.sum(self.passback[i-1],axis=[1,2,3]))
    
                                self.passback[i-2].assign(tf.matmul(self.passback[i-1], tf.transpose(self.weights[i-2])))
                                self.passback[i-2].assign(self.passback[i-2] * core.activate_backward(self.activations[0], self.z[i]))
                    else: 
                        self.del_w[i].assign(tf.matmul(tf.transpose(self.z[i]), self.passback[i+1]))
                        if self.use_b[i]:
                            self.del_b[i].assign(np.average(self.passback[i+1],axis=1))
                          
                        self.passback[i].assign(tf.matmul(self.passback[i+1], tf.transpose(self.weights[i])))
                        self.passback[i].assign(self.passback[i] * core.activate_backward(self.activations[0], self.z[i]))
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
        else:
            Error_Handler.model_cannot_be_updated_optimizier_not_set()


    def compute_tsne(self, data, labels, output_layer, batch_size, total_samples, tsne_save_file_path):
        try:
            raw_data = np.shape(self.z[output_layer][0,])
            
            batches = int(total_samples/batch_size)
            tsne_outputs = 0

            if len(raw_data) == 3:
                tsne_outputs = np.zeros(shape=(total_samples, raw_data[0], raw_data[1], raw_data[2]))
            elif len(raw_data) == 1:
                tsne_outputs = np.zeros(shape=(total_samples, raw_data[0]))

            for i in range(batches):
                BP_Model.forward(self, data[i*batch_size:(i+1)*batch_size], is_training=False)
                predictions = self.z[output_layer]

                tsne_outputs[i*batch_size:(i+1)*batch_size] = predictions.numpy()

            return tsne_outputs, labels
        except:
            print('Error while creating t-SNE in BP model.')

            
    def infer(self, data, labels, batch_size):
        samples = np.shape(data)[0]
        batches = int(samples/batch_size)
        
        if not Validator.validate_batch_size(samples, batches):
            Error_Handler.invalid_partial_batch()
        
        total_accuracy = 0
        
        for i in range(batches):
            BP_Model.forward(self, data[i*batch_size:(i+1)*batch_size], is_training=False)
            predictions = self.z[-1]
            predictions = tf.argmax(predictions, axis=1)
            accuracy = core.compute_accuracy(predictions, labels[i*batch_size:(i+1)*batch_size])
            total_accuracy += accuracy
        accuracy_percentage = ((total_accuracy/batches) * 100)
        return accuracy_percentage
    
    
