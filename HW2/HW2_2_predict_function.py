import numpy as np
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



class PredictFunction(MLAlgorithmTemplate):
    def __init__(self, learned_parameters=None, hyperparameters=None):
        super().__init__(hyperparameters)
        # 
        self.learned_parameters = learned_parameters
        self.parameters_assigned = learned_parameters is not None

    def predict(self, X):
        #Ensuring that parameters have been learned
        if not self.parameters_assigned:
            raise ValueError("Model parameters have not been learned yet.")
   
        intercept_column = np.ones((X.shape[0], 1))
        X_augmented = np.hstack((intercept_column, X))

        probabilities = self._predict_probability(X_augmented)

       
        predictions = (probabilities >= 0.5).astype(int)

        return predictions

    #  predict probability helper function
    def _predict_probability(self, X):
        print("Predicting: Probability of class label 0 .. ")
       
        return np.full(X.shape[0], 0.5)

# Example usage
if __name__ == "__main__":
   
    X_test = np.array([[4, 5], [6, 7], [8, 9]])

    learned_parameters = np.array([0.1, 0.2, 0.3])

 
    predict_func = PredictFunction(learned_parameters=learned_parameters)

    predictions = predict_func.predict(X_test)

    print("Predictions (class labels):", predictions)
