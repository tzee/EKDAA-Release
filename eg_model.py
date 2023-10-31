import os 
import numpy as np
import tensorflow as tf
from tensorflow import keras 

from library.FNN import FNN
from library.core import Core
from library.FNN_formatter import FNN_Formatter
from library.learning_mechs import Learning_Mechs
from library.data_preprocessor import Data_Preprocessor 

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

EPOCHS = 500
BATCH_SIZE = 50

FULL_TRAIN_SAMPLES = 50000
FULL_TEST_SAMPLES = 10000

TRAIN_SAMPLES = FULL_TRAIN_SAMPLES
TEST_SAMPLES  = FULL_TEST_SAMPLES

IMG_WIDTH    = 32
IMG_HEIGHT   = 32
IMG_CHANNELS = 3

core = Core()
data_preprocessor = Data_Preprocessor()

with tf.device('/gpu:0'):
    
    model = FNN("eg_model", 0)
    model.add_input(input_shape=(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)) 
    model.add_conv(filters=64, filter_size=3, activation='tanh', w_init='glorot_uniform', 
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_conv(filters=64, filter_size=3, activation='tanh', w_init='glorot_uniform',
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_maxpool(kernel=2, stride=2) 
    model.add_conv(filters=64, filter_size=3, activation='tanh', w_init='glorot_uniform',
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_conv(filters=128, filter_size=3, activation='tanh', w_init='glorot_uniform', 
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_maxpool(kernel=2, stride=2)   
    model.add_conv(filters=128, filter_size=3, activation='tanh', w_init='glorot_uniform',
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_conv(filters=128, filter_size=3, activation='tanh', w_init='glorot_uniform',
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_maxpool(kernel=2, stride=2)
  
    model.flatten()

    #example fully-connected model
    #model.add_input(input_shape=(BATCH_SIZE, IMG_WIDTH*IMG_HEIGHT*IMG_CHANNELS))
    #model.add_fc(nodes=2048, activation='tanh', w_init='glorot_uniform', 
    #             use_bias=True, b_init='glorot_uniform', dropout=0.24)
    #model.add_fc(nodes=1024, activation='tanh', w_init='glorot_uniform', 
    #             use_bias=True, b_init='glorot_uniform', dropout=0.28)

    model.add_fc(nodes=128, activation='tanh', w_init='glorot_uniform', 
                 use_bias=False, b_init='glorot_uniform', dropout=0.0)
    model.add_fc(nodes=10, activation='softmax', w_init='glorot_uniform', 
                 use_bias=False, b_init='glorot_uniform', dropout=0.0)
    
    model.print_model()

    #Set the learning rule 
    #model.build(learning_mechanism=Learning_Mechs.BP())
    model.build(learning_mechanism=Learning_Mechs.EKDAA())
    
    ###CIFAR-10
    cifar10 = keras.datasets.cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    train_x -= 0.5
    test_x -= 0.5
    
    #Needed for grayscale images
    #train_x = np.expand_dims(train_x, axis=3)
    #test_x = np.expand_dims(train_x, axis=3)
    
    train_x = train_x[0:TRAIN_SAMPLES]
    train_y = train_y[0:TRAIN_SAMPLES]
    
    test_x = test_x[0:FULL_TEST_SAMPLES]
    test_y = test_y[0:FULL_TEST_SAMPLES]
   
    opt = tf.optimizers.SGD(learning_rate=1e-3, momentum=0.85)
    model.set_optimizer(opt=opt)

    #Need to reshape for fully-connected model
    #train_x = np.reshape(train_x, (-1, 3072))
    
    accuracy = model.infer(data=train_x, labels=train_y, batch_size=BATCH_SIZE)
    formatted_accuracy = FNN_Formatter.format_accuracy(acc=accuracy, prec=2, epoch=0, use_epoch=False)
    print("Train Initial " + formatted_accuracy)
    
    for e in range(EPOCHS):
        shuffle_x, shuffle_y = data_preprocessor.shuffle(train_x, train_y)
        batches = int(TRAIN_SAMPLES / BATCH_SIZE)
        
        n_batches =0
        
        for b in range(batches):
            one_hot = core.convert_to_one_hot(shuffle_y[b*BATCH_SIZE:(b+1)*BATCH_SIZE], 10)
            
            model.forward(data=shuffle_x[b*BATCH_SIZE:(b+1)*BATCH_SIZE], is_training=True)
            model.backward(data=train_y, labels=one_hot)
            model.update()
            n_batches += 1
            print("\r {0} batches seen".format(n_batches), end="")
        print()
            
        accuracy = model.infer(data=train_x, labels=train_y, batch_size=BATCH_SIZE)
        formatted_accuracy = FNN_Formatter.format_accuracy(acc=accuracy, prec=2, epoch=e+1, use_epoch=True)
        print(formatted_accuracy)

        test_accuracy = model.infer(data=test_x, labels=test_y, batch_size=BATCH_SIZE)
        formatted_test_accuracy = FNN_Formatter.format_accuracy(acc=test_accuracy, prec=2, epoch=e+1, use_epoch=True)
        print(formatted_test_accuracy)
        
    #Final train accuracy
    train_accuracy = model.infer(data=train_x, labels=train_y, batch_size=BATCH_SIZE)
    formatted_train_accuracy = FNN_Formatter.format_accuracy(acc=train_accuracy, prec=2, epoch=e, use_epoch=False)
    print("Train Final " + formatted_train_accuracy)
    
    #Final test accuracy
    test_accuracy = model.infer(data=test_x, labels=test_y, batch_size=BATCH_SIZE)
    formatted_test_accuracy = FNN_Formatter.format_accuracy(acc=test_accuracy, prec=2, epoch=-1, use_epoch=False)
    print("Test Final " + formatted_test_accuracy)

    
    
