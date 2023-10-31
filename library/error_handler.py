import sys

class Error_Handler:
    
    #Static error messages
    ERROR_CAUSED_PROGRAM_TERMINATION      = "Error caused program termination."
    INVALID_LEARNING_MECHANISM            = "Learning mechanism not understood."
    INVALID_MODEL_STRUCTURE               = "The model design is not valid."
    INVALID_MODEL_EXE_MODEL_NOT_BUILT     =  "The model cannot be run before it has been built."
    MODEL_CANNOT_BE_UPDATED_NO_OPT        = "Model cannot be updated. Optimizer has not been set."
    INVALID_PARTIAL_BATCH                 = "BATCH_SIZE must evenly divide batches. Partial batches is not supported."
    UNKNOWN_ERROR_IN_FORWARD              = "An unknown error has occured in the forward propagation."
    UNKNOWN_ERROR_IN_BACKWARD             = "An unknown error has occured in the backward propagation."
    FORWARD_ACTIVATION_TYPE_INVALID       = "The forward activation function is not understood."
    BACKWARD_ACTIVATION_TYPE_INVALID      = "The backward activation function is not understood."
    ERROR_COMP_CROSS_ENTROPY_LOSS         = "An error occured computing the cross entropy loss."
    ERROR_COMP_CROSS_ENTROPY_LOSS_DIR     = "An error occured computing the cross entropy loss derivative."
    WEIGHT_INIT_TYPE_INVALID              = "Weight initialization type is not understood."
    UNKNOWN_LEARN_MECH_CANNOT_SET_OPT     = "Cannot set optimizer, learning mechanism is unknown."
    UNKNOWN_LEARN_MECH_CANNOT_COMP_FOR    = "Cannot compute forward, learning mechanism is unknown."
    UNKNOWN_LEARN_MECH_CANNOT_COMP_BACK   = "Cannot compute backward, learning mechanism is unknown."
    UNKNOWN_LEARN_MECH_CANNOT_COMP_UP     = "Cannot compute update, learning mechanism is unknown."
    UNKNOWN_LEARN_MECH_CANNOT_COMP_INFER  = "Cannot compute infer, learning mechanism is unknown."
    MAXPOOL_CAN_ONLY_BE_COMPLETED_ON_CONV = "Maxpool can only be completed on a convolutional layer."
    
    #Static warning messages
    WARNING_ONLY                          = "Warning: "
    BP_DOES_NOT_SUPPORT_SETTING_BETA      = WARNING_ONLY + "Back-prop does not support the beta parameter."
    BP_DOES_NOT_SUPPORT_SETTING_GAMMA     = WARNING_ONLY + "Back-prop does not support the gamma parameter."
    LEARN_MECH_DOES_NOT_SUPPORT_SET_BETA  = WARNING_ONLY + "Learning mechanism does not support beta."
    LEARN_MECH_DOES_NOT_SUPPORT_SET_GAMMA = WARNING_ONLY + "Learning mechanism does not support beta."

    #Not implemented warning messages
    IMPLEMENTATION_WARNING                = "Feature not implemented: "
    MAXPOOL_PARAMS_NOT_IMPLEMENTED        = IMPLEMENTATION_WARNING + "maxpool currently does not support custom kernel or stride"

    #Errors 
    def invalid_learning_mechanism():
        print(Error_Handler.INVALID_LEARNING_MECHANISM)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def invalid_model_structure_to_build():
        print(Error_Handler.INVALID_MODEL_STRUCTURE)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def invalid_model_execution_model_not_built():
        print(Error_Handler.INVALID_MODEL_EXE_MODEL_NOT_BUILT)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
  
    
    def model_cannot_be_updated_optimizier_not_set():
        print(Error_Handler.MODEL_CANNOT_BE_UPDATED_NO_OPT)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def invalid_partial_batch():
        print(Error_Handler.INVALID_PARTIAL_BATCH)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return    
    
    
    def unknown_error_in_forward():
        print(Error_Handler.UNKNOWN_ERROR_IN_FORWARD)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def unknown_error_in_backward():
        print(Error_Handler.UNKNOWN_ERROR_IN_BACKWARD)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def unknown_forward_activation_function():
        print(Error_Handler.FORWARD_ACTIVATION_TYPE_INVALID)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def unknown_backward_activation_function():
        print(Error_Handler.BACKWARD_ACTIVATION_TYPE_INVALID)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def error_computing_cross_entropy_loss():
        print(Error_Handler.ERROR_COMP_CROSS_ENTROPY_LOSS)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def error_computing_cross_entropy_loss_derivative():
        print(Error_Handler.ERROR_COMP_CROSS_ENTROPY_LOSS_DIR)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return
    
    
    def weight_initialization_type_not_understood():
        print(Error_Handler.WEIGHT_INIT_TYPE_INVALID)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return    
    
    
    def unknown_learning_mechanism_cannot_set_optimizer():
        print(Error_Handler.UNKNOWN_LEARN_MECH_CANNOT_SET_OPT)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return    
    
    
    def unknown_learning_mechanism_cannot_compute_forward():
        print(Error_Handler.UNKNOWN_LEARN_MECH_CANNOT_COMP_FOR)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return   
    
    
    def unknown_learning_mechanism_cannot_compute_backward():
        print(Error_Handler.UNKNOWN_LEARN_MECH_CANNOT_COMP_BACK)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return    
    
    
    def unknown_learning_mechanism_cannot_compute_update():
        print(Error_Handler.UNKNOWN_LEARN_MECH_CANNOT_COMP_UP)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return    
    
    
    def unknown_learning_mechanism_cannot_compute_infer():
        print(Error_Handler.UNKNOWN_LEARN_MECH_CANNOT_COMP_INFER)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return    
    
    
    def maxpool_can_only_be_completed_on_convolutional_layer():
        print(Error_Handler.MAXPOOL_CAN_ONLY_BE_COMPLETED_ON_CONV)
        print(Error_Handler.ERROR_CAUSED_PROGRAM_TERMINATION)
        sys.exit()
        return    
    
    
    #Warnings
    def backprop_does_not_support_beta():
        print(Error_Handler.BP_DOES_NOT_SUPPORT_SETTING_BETA)
        return    
  
    
    def backprop_does_not_support_gamma():
        print(Error_Handler.BP_DOES_NOT_SUPPORT_SETTING_GAMMA)
        return    
    
    
    def learning_mech_does_not_support_beta():
        print(Error_Handler.LEARN_MECH_DOES_NOT_SUPPORT_SET_BETA)
        return
    
    
    def learning_mech_does_not_support_gamma():
        print(Error_Handler.LEARN_MECH_DOES_NOT_SUPPORT_SET_GAMMA)
        return


    #Feature not implemented warnings
    def maxpool_parameters_cannot_be_custom():
        print(Error_Handler.MAXPOOL_PARAMS_NOT_IMPLEMENTED)
        return    
 
