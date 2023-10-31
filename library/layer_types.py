class Layer_Types:
    
    _Input   = "input"
    _Conv    = "conv"
    _Maxpool = "maxpool"
    _Flatten = "flatten"
    _FC      = "fc"
    
    #Get methods only
    def Input():
        return Layer_Types._Input
    
    def Conv():
        return Layer_Types._Conv
    
    def Maxpool():
        return Layer_Types._Maxpool
    
    def Flatten():
        return Layer_Types._Flatten
    
    def FC():
        return Layer_Types._FC
    
    
