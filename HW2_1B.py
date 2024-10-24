# 1B.py


# MLAlgorithmTemplate class 
class MLAlgorithmTemplate:
    def __init__(self, hyperparameters=None):
        #  when parameters are assigned 
        self.parameters_assigned = False
       
        self.learned_parameters = None  #  learned parameters
        #  hyperparameters 
        self.hyperparameters = hyperparameters
    
    # Training function 
    def train(self, X, y):
        raise NotImplementedError("Subclasses must implement this function.  IMPLMENT ERROR!")
    
    
    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this function.  IMPLMENT ERROR!")
    
   
    def get_parameters_assigned(self):
        return self.parameters_assigned
    
    def get_learned_parameters(self):
        return self.learned_parameters
    
    def get_hyperparameters(self):
        return self.hyperparameters
    
    # update  parameters 
    def _update_learned_parameters(self, parameters):
        self.learned_parameters = parameters
        self.parameters_assigned = True


# Test the class
if __name__ == "__main__":
    # Instantiate the class
    ml_template = MLAlgorithmTemplate(hyperparameters={"learning_rate": 0.01})
    
    # Print hyperparameters to check initialization
    print("Hyperparameters:", ml_template.get_hyperparameters())
    
    # Try calling train to see if it raises NotImplementedError
    try:
        ml_template.train(None, None)
    except NotImplementedError as e:
        print(e)
