import numpy as np
import tensorflow as tf

from library.loss import Loss
from library.activation import Activation
from library.normalization import Normalization
from library.weight_initialization import Weight_Initialization

loss = Loss()
activation = Activation()
normalization = Normalization
weight_initialization = Weight_Initialization


class Core:
    
    #Helper functions 
    def convert_to_one_hot(self, labels, num_classes):
        labels = tf.cast(labels, tf.int64)
        return tf.one_hot(labels, num_classes)
    
    
    def pascanu_rescaling(self, matrix, epsilon, rho):
        return normalization.pascanu_rescaling(self, matrix, epsilon, rho)

    
    #Loss functions 
    def softmax_loss(self, predictions, labels):
        print(np.sum(predictions-labels))
        return predictions - labels


    def cross_entropy_loss(self, predictions, labels, classes):
        return tf.nn.softmax_cross_entropy_with_logits(labels, predictions)

    
    def cross_entropy_loss_batch(self, predictions, labels, classes):
        loss = 0 
        for i in range(len(labels)):
            loss += tf.nn.softmax_cross_entropy_with_logits(labels[i], predictions[i])
        return loss
    
  
    def cross_entropy_loss64(self, predictions, labels, classes):
        return loss.cross_entropy_loss64(predictions, labels, classes)
            
    
    def cross_entropy_loss_derivative(self, predictions, labels, classes):
        return loss.cross_entropy_loss_derivative(predictions, labels, classes)
        
    
    def cross_entropy_loss_derivative64(self, predictions, labels, classes):
        return loss.cross_entropy_loss_derivative64(predictions, labels, classes)
         
    
    #Convolutional functions
    def maxpool(self, data):
        result = tf.compat.v2.nn.max_pool2d(input=data, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
        return result
        
    def pool_upsample(self, pooled_image, window_size, stride, padding):
        result = tf.keras.layers.UpSampling2D(size=(window_size,window_size), interpolation='bilinear')(pooled_image)        
        return result

    #Conv 1st iteration used sliding window implementation
    #Conv 2nd iteration used a Topelitz matrix, this was quick but limited number of layers/filters
    #Conv 3rd iteration used Fast Fourier Transform
    #Currently using Tensorflow's implementation as it is signficantly quicker but,
    #still quite slower than defining a full tensorflow graph and using in-built training,
    #Better convolution could be done here, especially with a CUDA implementation.
    def conv(self, x, kernel):
        return tf.compat.v2.nn.conv2d(input=x, filters=kernel, strides=[1,1,1,1], padding='SAME')
        

    #Match convolution implementation here
    #Normalizing by filter shape to manage error singal size
    #Tensorflow's conv2d_backprop_filter computes: input conv delta_output = delta_filter
    def deconv(self, del_w, idx, x, kernel, out_filters, filter_size):    
        filters =  tf.compat.v1.nn.conv2d_backprop_filter(input=x, filter_sizes=np.shape(del_w[idx]), out_backprop=kernel, strides=[1,1,1,1], padding="SAME", data_format='NHWC', dilations=[1,1,1,1])
        
        if np.max(filters) > 1:
              filters = tf.divide(filters, np.max(filters))
        filters = tf.divide(filters, np.shape(del_w[idx])[2])
        filters = tf.divide(filters, np.shape(del_w[idx])[3])
    
        return filters
        
    #Match convolution implementation here
    #If using regular convolution, make sure to use the rotate_180 function so that
    #in the forward pass the model is computing correlation, and the backward pass
    #is computing convolution
    #Tensorflow's conv2d_backprop_input computes: delta_output conv flip(filters) = delta_input
    #If swapping function, make sure to follow this formula
    def deconv_pb(self, desired_input, x, kernel):
        x = tf.compat.v1.nn.conv2d_backprop_input(input_sizes=np.shape(desired_input), filter=kernel, out_backprop=x, strides=[1,1,1,1], padding="SAME", data_format='NHWC', dilations=[1,1,1,1])

        if np.max(x) > 1:
            x = tf.divide(x, np.max(x))
            
        return x

    
    def rotate_180(self, x):
        x = np.flip(x, 0)
        x = np.flip(x, 1)
        return x


    #Activation                                           
    def activate(self, act_type, layer):
        return activation.forward_activation(act_type, layer)
    
    
    def activate_backward(self, act_type, layer):
        return activation.backward_activation(act_type, layer)


    #Weights/biases
    def set_weights(self, weight_init, shape):
        return weight_initialization.build_weights(self, weight_init, shape)
    
    
    def set_biases(self, bias_init, shape):
        return weight_initialization.build_weights(self, bias_init, shape)
    
    
    #Metrics
    def compute_accuracy(self, predicted, actual):
        correct = 0 
        
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                correct += 1

        return (correct / len(actual))
    
    
    def compute_disorder(self, loss, e):
        disorder = loss
                       
        for i in range(len(e)):
            error_kernel = e[i]
                        
            if tf.shape(error_kernel)[0] == 2:
                for n in range(tf.shape(error_kernel)[0]):      
                    disorder += np.sum((error_kernel[n] * np.transpose(error_kernel[n])))
            else:
                for n in range(tf.shape(error_kernel)[0]):                    
                    for c in range(tf.shape((error_kernel[n]))[0]):
                        disorder += np.sum((error_kernel[n][c] * np.transpose(error_kernel[n][c])))
        return disorder
