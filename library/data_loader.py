import pickle
import numpy as np
import scipy.io as sio

from library.error_handler import Error_Handler

class Data_Loader:
    
    def load_data_from_pkl(self, filepath_x, filepath_y, ordering="True"):
        with open(filepath_x, "rb") as file_x:
            x_data = pickle.load(file_x)
            
        with open(filepath_y, "rb") as file_y:
            y_data = pickle.load(file_y)
            
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        
        if np.min(y_data) > 0:
            y_data = y_data - np.min(y_data)
                    
        reordered_data = Data_Loader.__reorder(x_data, ordering)
    
        return reordered_data, y_data
    
    
    def load_data_from_npy(self, filepath_x, filepath_y, ordering="True"):
        x_data = np.load(filepath_x)
        y_data = np.load(filepath_y)
        
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        
        if np.min(y_data) > 0:
            y_data = y_data - np.min(y_data)
                    
        reordered_data = Data_Loader.__reorder(x_data, ordering)
    
        return reordered_data, y_data
    
    
    def load_data_from_mat(self, filepath, x_key, y_key, ordering):
        mat_dict = sio.loadmat(filepath)
        x_data = mat_dict[x_key]
        y_data = mat_dict[y_key]
        
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        
        if np.min(y_data) > 0:
            y_data = y_data - np.min(y_data)
                    
        reordered_data = Data_Loader.__reorder(x_data, ordering)
    
        return reordered_data, y_data
    
    
    def __reorder(x, ordering):
        if ordering == "SWHC":
            return x
        elif ordering == "CWHS":
             x = np.swapaxes(x, 3, 0)
             return x
        elif ordering == "WHCS":
             x = np.rollaxis(x, 2, 0)
             x = np.swapaxes(x, 0, 3)
             return x         
        elif ordering == "WHSC":
             x = np.rollaxis(x, 3, 0)
             x = np.swapaxes(x, 0, 3)
             return x
        elif ordering == "SCWH":
             x = np.rollaxis(x, 1, 4)
             return x      
        elif ordering == "CSWH":
             x = np.swapaxes(x, 0, 1)
             x = np.rollaxis(x, 1, 4)
             return x
        else:
            Error_Handler.error_in_data_ordering()
