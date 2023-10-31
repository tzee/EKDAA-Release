class Z:
    
    #Activation functions
    _elu        = "elu"
    _leaky_relu = "leaky_relu"
    _relu       = "relu" 
    _relu6      = "relu6"
    _sigmoid    = "sigmoid"
    _signum     = "signum"
    _softmax    = "softmax"
    _softplus   = "softplus"
    _tanh       = "tanh"
    
    #Weight initializations
    _glorot_uniform = "glorot_uniform"
    _glorot_normal = "glorot_normal"
    
    #Pooling types 
    _pmax = "pmax"
    _pave = "pave"
    
    #Data segments
    _train = "train"
    _valid = "valid"
    _test = "test"
    
  
    #Get methods only 
    #Activation functions 
    def elu():
        return Z._elu

    
    def leaky_relu():
        return Z._leaky_relu

    
    def relu():
        return Z._relu

    
    def relu6():
        return Z._relu6

    
    def sigmoid():
        return Z._sigmoid

    
    def signum():
        return Z._signum

    
    def softmax():
        return Z._softmax

    
    def softplus():
        return Z._softplus

    
    def tanh():
        return Z._tanh

    
    #Weight initailizations
    def glorot_uniform():
        return Z._glorot_uniform

    
    def glorot_normal():
        return Z._glorot_normal

    
    #Pooling types
    def pmax():
        return Z._pmax

    
    def pave():
        return Z._pave
    
    
    #Data segments
    def train():
        return Z._train

    
    def valid():
        return Z._valid

    
    def test():
        return Z._test
    
