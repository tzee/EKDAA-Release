import tensorflow as tf

class Normalization:

    def pascanu_rescaling(self, matrix, epsilon, rho):
        #Compute frobenius norm
        matrix_T = tf.transpose(matrix)

        #Rescales the entire matrix
        #If we wanted to rescale row-wise we should just add dim=0
        frob_norm = tf.norm(matrix_T) + epsilon #Grad stability

        #If the norm is bigger than the threshold, pascanu rescaling should take place
        #Otherwise do not scale
        avg_frob_norm = (tf.reduce_sum(tf.reshape(frob_norm, [-1]), 0) / len(tf.reshape(frob_norm, [-1])))

        if avg_frob_norm > rho:
            matrix_T /= frob_norm
            matrix_T *= rho
            matrix = tf.transpose(matrix_T)

        return matrix
