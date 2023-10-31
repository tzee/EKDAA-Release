class Validator:
    
    def valid_model_structure(self, model):
        print("Model validation to be implemented...")
        return True

    
    def validate_batch_size(data, batch_size):
        temp = (data / batch_size)
        if temp.is_integer():
            return True
        return False
