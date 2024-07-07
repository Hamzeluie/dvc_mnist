import numpy as np

class TrainBaseModule:
    
    def __train_val_test_dataloader(self):
        pass
    
    def __set_parameters(self):
        pass
    
    def __train_one_step(self):
        pass
    
    def __train_one_epoch(self):
        pass
    
    def train(self):
        pass
    
    def validation_evaluation(self):
        pass
    
    def test_evaluation(self):
        pass


class InferenceBaseModule:
    
    def __set_parameters(self):
        pass
    
    def inference(self, image:np.array)->np.array:
        pass
    
    def batch_inference(self, input_dir:str, file_extention:str, output_path:str):
        pass