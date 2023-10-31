# EKDAA with zFlow

## About
zFlow is a machine learning library with the aim of being able to compare 
different learning mechanisms quickly and easy in an apples-to-apples fashion. zFlow allows for defining a learning rule agnostic model before deciding on the learning rule to train the graph. This allows for easily swapping what learning rule will be used to train a model. 

EKDAA, introduced in "A Robust Backpropagation-Free Framework for Images" by Zee et al. (https://openreview.net/forum?id=leqr0vQzeN) is a bio-plausible learning mechanism for training images with learning error kernels propagate error signals locally. EKDAA shares the same forward pass as the equivalent BP model, but the backward pass differs significantly. 

zFlow currently supports back-propagation and EKDAA as learning rules. Because EKDAA backward's pass is completely different than back-propagation, it requires different matrices to propagate error signal. zFlow models only allocate memory and build required forward and backward structures after the build function is called, and a parameter of the build function is the learning mechanism desired. Therefore, even though BP and EKDAA have entirely different model structures, it is easy to build a comparable model without needing to declare the intricacies with each learning rule.   

zFlow is made to easily adapt new learning rules to allow further apples-to-apples comparison of alternative learning rules to BP. 

## Examples

Defining a convolutional neural network in zFlow:
```model = FNN("eg_model", 0)
    model.add_input(input_shape=(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)) 
    model.add_conv(filters=64, filter_size=3, activation='tanh', w_init='glorot_uniform', 
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_conv(filters=64, filter_size=3, activation='tanh', w_init='glorot_uniform',
                   use_bias=False, b_init='glorot_uniform', dropout=0.0)  
    model.add_maxpool(kernel=2, stride=2) 
    model.flatten()
    model.add_fc(nodes=128, activation='tanh', w_init='glorot_uniform', 
                 use_bias=False, b_init='glorot_uniform', dropout=0.0)
    model.add_fc(nodes=10, activation='softmax', w_init='glorot_uniform', 
                 use_bias=False, b_init='glorot_uniform', dropout=0.0)
    
    model.print_model()

```

Building the model and allocating it into memory for BP or EKDAA: 
```
model.build(learning_mechanism=Learning_Mechs.BP())
model.build(learning_mechanism=Learning_Mechs.EKDAA())
```

Creating a simple training loop in zFlow:
```
for e in range(EPOCHS):
    shuffle_x, shuffle_y = data_preprocessor.shuffle(train_x, train_y)
    batches = int(TRAIN_SAMPLES / BATCH_SIZE)
        
    n_batches = 0
        
    for b in range(batches):
        one_hot = core.convert_to_one_hot(shuffle_y[b*BATCH_SIZE:(b+1)*BATCH_SIZE], 10)
            
        model.forward(data=shuffle_x[b*BATCH_SIZE:(b+1)*BATCH_SIZE], is_training=True)
            model.backward(data=train_y, labels=one_hot)
            model.update()
            n_batches += 1
            print("\r {0} batches seen".format(n_batches), end="")
        print()
            
```

Simple inference in zFlow:
```
accuracy = model.infer(data=train_x, labels=train_y, batch_size=BATCH_SIZE)
formatted_accuracy = FNN_Formatter.format_accuracy(acc=accuracy, prec=2, epoch=0, use_epoch=False)
print("Train Initial " + formatted_accuracy)
```
## Important Notes
The following list are important points to keep in mind when using the current iteration on zFlow for model development:

* A batch size must be set such that dividing the total train and test samples by the batch size creates an even number of full batches. zFlow cannot currently suport partial batches.
* Maxpooling is currently restricted to only a kernel of [2, 2] with a stride of 2. 
* There is currently no validation in place to ensure that a defined model can actually exist. zFlow is set up for validation, so custom validation can be done by modifying validator.py to check the feasibility of a defined model before memory is allocated for it.
* Maxpooling will only work directly after a convolutional layer is defined. This is due to the way the graph is created in zFlow.
* Dropout is not implemented on the backend. Dropout is a parameter for defining a model, but dropout will not be computed on the backend. 


## Citation

Please cite our paper if it is helpful to your work:
```
@article{

title={A Robust Backpropagation-Free Framework for Images},
author={Zee, Timothy and Ororbia, Alexander G and Mali, Ankur and Nwogu, Ifeoma},
journal={Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=leqr0vQzeN},
note={}
}
```

