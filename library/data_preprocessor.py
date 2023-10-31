import cv2
import imutils
import numpy as np
from PIL import Image
import tensorflow as tf

from library.error_handler import Error_Handler


class Data_Preprocessor:
    
    def shuffle(self, images, labels):
        batch_len = images.shape[0]
        random_ordering = np.random.permutation(batch_len)
        
        ordered_images = np.zeros(np.shape(images), dtype=np.float32)
        ordered_labels = np.zeros(np.shape(labels), dtype=np.uint8)
        
        for i in range(batch_len):
            ordered_images[i] = images[random_ordering[i]]
            ordered_labels[i] = labels[random_ordering[i]]
        
        return ordered_images, ordered_labels


    def batchify_and_shuffle(self, dataset, labels, batch_size):
        samples, width, height, channels = tf.shape(dataset)
        batches = int(np.ceil(samples/batch_size))
        
        last_batch_size = samples - (batch_size * (batches-1))
        
        random_data = tf.Variable(tf.zeros(tf.shape(dataset)), dtype=tf.float32)
        random_labels = tf.Variable(tf.zeros(tf.shape(labels)), dtype=tf.float32)
        
        #Shuffle the data
        random_ordering = tf.constant(value=np.random.permutation(samples.numpy()), dtype=tf.int32)
        
        for i in range(samples):
            random_data[i].assign(dataset[random_ordering[i]])
            random_labels[i].assign(labels[random_ordering[i]])

        result_data = tf.Variable(tf.zeros(shape=[batches, batch_size, width, height, channels]), dtype=tf.float32)
        result_labels = tf.Variable(tf.zeros(shape=[batches, batch_size]), dtype=tf.float32)

        for i in range(batches):
            if i == (batches - 1):
                result_data[i, 0:last_batch_size].assign(random_data[i*batch_size:(i*batch_size)+last_batch_size])
                result_labels[i, 0:last_batch_size].assign(random_labels[i*batch_size:(i*batch_size)+last_batch_size])
            else:
                result_data[i].assign(random_data[i*batch_size:(i+1)*batch_size])
                result_labels[i].assign(random_labels[i*batch_size:(i+1)*batch_size])
            
        #Current shape of the data
        #[batches, batch_size, width, height, channels]
                
        #Required shape of the data
        #[batches, batch_size, channels, width, height]
        result_data = tf.Variable(np.rollaxis(result_data.numpy(), 4, 2), dtype=tf.float32)

        result_labels = tf.cast(result_labels, dtype=tf.int64)
    
        return result_data, result_labels
        
  
    def batchify_data(self, dataset, batch_size):
        samples, width, height, channels = np.shape(dataset)
        batches = int(np.ceil(samples/batch_size))
        
        last_batch_size = samples - (batch_size * (batches-1))
        
        result = tf.Variable(tf.zeros(shape=[batches, batch_size, width, height, channels], dtype=tf.float32))
        
        for i in range(batches):
            if i == (batches - 1):
                result[i, 0:last_batch_size].assign(dataset[i*batch_size:(i*batch_size)+last_batch_size])
            else:
                result[i].assign(dataset[i*batch_size:(i+1)*batch_size])

        #Current shape of the data
        #[batches, batch_size, width, height, channels]
                
        #Required shape of the data
        #[batches, batch_size, channels, width, height]
        result_final = tf.Variable(tf.transpose(result, perm=[0, 1, 4, 2, 3]))
        
        return result_final


    def batchify_labels(self, labels, batch_size):
        samples = tf.size(labels)
        
        batches = int(np.ceil(samples/batch_size))
        
        last_batch_size = samples - (batch_size * (batches-1))
        
        result = tf.Variable(tf.zeros(shape=[batches, batch_size], dtype=tf.float32))
        
        for i in range(batches):
            if i == (batches - 1):
                result[i, 0:last_batch_size].assign(labels[i*batch_size:(i*batch_size)+last_batch_size])
            else:
                result[i].assign(labels[i*batch_size:(i+1)*batch_size])
                
        result = tf.cast(result, dtype=tf.int64)

        return result


    def visualize_image(self, data, filepath):
        data = data.astype(np.uint8)
        img = Image.fromarray(data, 'RGB')
        img.save(filepath + '.png')
        img.show()


    def standardize_data(self, x, y):
        #x = train, y = test
        mean_x = np.mean(x)
        std_x = np.std(x)
        
        x = x - mean_x
        x = x / std_x
        
        y = y - mean_x
        y = y / std_x
        
        return x, y

    def normalize_data(self, x, y):
        #x = train, y = test
        min_x = np.min(x)
        max_x = np.max(x)
        
        x = (x - min_x)
        x = (x / (max_x - min_x))

        y = (y - min_x)
        y = (y - (max_x - min_x))
        
        return x, y
    
    def ZCA_whiten_data(self, x, y):
        #x = train, y = test
        x_flat = np.reshape(x, newshape=(np.shape(x)[0], np.shape(x)[1] * np.shape(x)[2] * np.shape(x)[3]))
        y_flat = np.reshape(y, newshape=(np.shape(y)[0], np.shape(y)[1] * np.shape(y)[2] * np.shape(y)[3]))
       
        x_flat= x_flat - x_flat.mean(axis=0)
        x_flat = x_flat / np.sqrt((x_flat ** 2).sum(axis=1))[:,None]
        
        y_flat = y_flat - y_flat.mean(axis=0)
        y_flat = y_flat / np.sqrt((y_flat ** 2).sum(axis=1))[:,None]
        
        cov_x = np.cov(x_flat, rowvar=True)
        cov_y = np.cov(y_flat, rowvar=True)
        
        U_x, S_x, V_x = np.linalg.svd(cov_x)  
        U_y, S_y, V_y = np.linalg.svd(cov_y)    

        epsilon = 1e-4
        zca_x = np.dot(U_x, np.dot(np.diag(1.0/np.sqrt(S_x + epsilon)), U_x.T))
        zca_y = np.dot(U_y, np.dot(np.diag(1.0/np.sqrt(S_y + epsilon)), U_y.T))
    
        zca_dot_x = np.dot(zca_x, x_flat)    
        zca_dot_y = np.dot(zca_y, y_flat)    
                
        min_zca_x, max_zca_x = np.min(zca_dot_x), np.max(zca_dot_x)
        min_zca_y, max_zca_y = np.min(zca_dot_y), np.max(zca_dot_y)
        
        norm_zca_x = zca_dot_x + (min_zca_x * -1)
        norm_zca_x = norm_zca_x / (max_zca_x - min_zca_x)
        norm_zca_x *= 255
        
        norm_zca_y = zca_dot_y + (min_zca_y * -1)
        norm_zca_y = norm_zca_y / (max_zca_y - min_zca_y)
        norm_zca_y *= 255
        
        norm_zca_x_r = np.reshape(norm_zca_x, newshape=(np.shape(x)))
        norm_zca_y_r = np.reshape(norm_zca_y, newshape=(np.shape(y)))

        norm_zca_x_r = np.uint8(norm_zca_x_r)
        norm_zca_y_r = np.uint8(norm_zca_y_r)

        return norm_zca_x_r, norm_zca_y_r
      
    
    def augment_data(self, x, augmentation_factor, use_crop_factor, min_rotation, 
                     max_rotation, h_flip_prob, v_flip_prob):
        CROP_FACTORS = [1.0, 0.8, 0.6]
        resolution = np.shape(x)[1] #width/height
        
        samples, width, height, channels = np.shape(x)
        
        pixels_08 = int(np.ceil((resolution * 0.8)))
        pixels_06 = int(np.ceil((resolution * 0.6)))
        
        pixel_08_offset = int(np.ceil((resolution - pixels_08)/2))
        pixel_06_offset = int(np.ceil((resolution - pixels_06)/2))
           
        y = np.zeros(shape=(samples*(augmentation_factor+1), width, height, channels))
        y[0:len(x)] = x
        
        try:
            if min_rotation < 0 or min_rotation > 360 or max_rotation < 0 or \
            max_rotation > 360 or h_flip_prob < 0 or h_flip_prob > 1 or \
            v_flip_prob < 0 or v_flip_prob > 1: 
                Error_Handler.error_in_augment_data()
            else:
                for i in range(samples):
                    for a in range(augmentation_factor):
                        #figure out 
                        if use_crop_factor:
                            crop_factor = np.random.choice(CROP_FACTORS, 1, True)
                        else:
                            crop_factor = 1.0
                            
                        rotation = np.random.random_integers(min_rotation, max_rotation)
                        horizontal_flip = np.random.binomial(1, h_flip_prob, 1)
                        vertical_flip = np.random.binomial(1, v_flip_prob, 1)
                    
                        y[(a+1)*samples + i] = x[i] 
                        
                        #Crop
                        if crop_factor == 0.8:
                            left_offset = np.random.randint(0, pixel_08_offset)
                            down_offset = np.random.randint(0, pixel_08_offset)
                            
                            y[(a+1)*samples + i] = cv2.resize(x[i][left_offset:left_offset+pixels_08,
                                                                   down_offset:down_offset+pixels_08], 
                                                             (32, 32), interpolation=cv2.INTER_AREA)
                        elif crop_factor == 0.6:
                            left_offset = np.random.randint(0, pixel_06_offset)
                            down_offset = np.random.randint(0, pixel_06_offset)
                             
                            y[(a+1)*samples + i] = cv2.resize(x[i][left_offset:left_offset+pixels_06,
                                                                   down_offset:down_offset+pixels_06],
                                                             (32, 32), interpolation=cv2.INTER_AREA)
                            
                        #Rotation
                        y[(a+1)*samples + i] = cv2.resize(imutils.rotate_bound(y[(a+1)*samples + i], rotation),
                                                         (32, 32), interpolation=cv2.INTER_AREA)
                        
                        #Horizontal Flip
                        if horizontal_flip:
                            y[(a+1)*samples + i] = cv2.flip(src=y[(a+1)*samples + i], flipCode=0)
                              
                        #Vertical Flip
                        if vertical_flip:
                            y[(a+1)*samples + i] = cv2.flip(src=y[(a+1)*samples + i], flipCode=1)                 
            return y
        except:
            Error_Handler.error_in_augment_data()
        return 
    
