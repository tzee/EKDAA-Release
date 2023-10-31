import numpy as np
import tensorflow as tf

from library.activation import Activation
from library.error_handler import Error_Handler

activation = Activation()

class Loss:

    def cross_entropy_loss(self, predictions, labels, classes):
        labels = tf.cast(labels, tf.float32)
        predictions = tf.cast(labels, tf.float32)

        if(len(labels) == len(predictions)):
            loss = tf.zeros(shape=[1, classes], dtype=tf.float32)
            for i in range(len(labels)):
                loss -= labels[i] * tf.math.log(predictions[i])
            result = tf.reduce_sum(loss, 1)
            return result
        else:
            Error_Handler.error_computing_cross_entropy_loss()
            
            
    def cross_entropy_loss_derivative(self, predictions, labels, classes):
        if(len(labels) == len(predictions)):
            return -labels/predictions
        else:
            Error_Handler.error_computing_cross_entropy_loss_derivative()


    def softmax_loss(self, predictions, labels):
        return predictions - labels
