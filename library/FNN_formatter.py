class FNN_Formatter:
    
    def format_input_layer(input_params):
        layer_type  = input_params[0]
        input_shape = str(input_params[1])
        
        format_layer = layer_type + " {shape: " + input_shape + "}"
        
        return format_layer
    
    
    def format_conv_layer(conv_params):
        layer_type   = conv_params[0]
        layer_number = str(conv_params[1])
        filters      = str(conv_params[2])
        filter_size  = str(conv_params[3])
        activation   = conv_params[4]
        init         = conv_params[5]
        dropout      = str(conv_params[8])
        
        
        format_layer = layer_type + "_" + layer_number + " {filters: " + \
                       filters + " [" + filter_size + "x" + \
                       filter_size + "]," + " activation: '" +  activation + \
                       "', init: '" + init + "', dropout: '" + dropout + "'}" 
        return format_layer
    
    
    def format_mp_layer(mp_params):
        layer_type   = mp_params[0]
        layer_number = str(mp_params[1])
        kernel       = str(mp_params[2])
        stride       = str(mp_params[3])
        
        format_layer = layer_type + "_" + layer_number + \
                       " {kernel: [" + kernel + "x" + kernel \
                        + "], stride: " + stride + "}"
        return format_layer
    
    
    def format_fc_layer(fc_params):
        layer_type   = fc_params[0]
        layer_number = str(fc_params[1])
        nodes        = str(fc_params[2])
        activation   = fc_params[3]
        init         = fc_params[4]
        dropout      = str(fc_params[7])
                
        format_layer = layer_type + "_" + layer_number + " {nodes: " + \
                       nodes + ", activation: '" + activation + \
                       "', init: '" + init + "' dropout: '" + dropout + "'}"       
        return format_layer

    
    def format_accuracy(acc, prec, epoch, use_epoch):
        trimmed_accuracy = round(acc, prec)
        if use_epoch:
            formatted_accuracy = "Epoch: " + str(epoch) + ", Accuracy: " + str(trimmed_accuracy) + "%"
        else:
            formatted_accuracy = "Accuracy: " + str(trimmed_accuracy) + "%"
        return formatted_accuracy
