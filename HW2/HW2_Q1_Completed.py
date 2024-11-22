# Import necessary packages
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

# LinearRegression class
class LinearRegression(MLAlgorithmTemplate):
    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)

    
    def train(self, X, y):
        # check that the hypparam are provided/fetched
        if not self.hyperparameters:
            raise ValueError("Hyperparameters are needed, ERROR ! REQS!")
        
        # Retrieve the hyperparameter
        variant = self.hyperparameters.get("variant", "MLE").upper()
        
        # Initialize σ² and b² 
        sigma_sqrd = 0
        b_sqrd = 1
        
        # Determine the variant type
        if variant == "MAP":
            # Check if required hyperparameters are provided
            if "sigma_sqrd" not in self.hyperparameters or "b_sqrd" not in self.hyperparameters:
                raise ValueError("MAP variant requires 'sigma_sqrd' and 'b_sqrd' hyperparameters to be present")
            sigma_sqrd = self.hyperparameters["sigma_sqrd"]
            b_sqrd = self.hyperparameters["b_sqrd"]
        elif variant == "MLE":
            # MLE default values 
            sigma_sqrd = 0
            b_sqrd = 1
        elif variant == "REGULARIZATION":

            if "sigma_sqrd" not in self.hyperparameters or "lambda" not in self.hyperparameters:
                raise ValueError("Regularization variant requires 'sigma_sqrd' and 'lambda' hyperparameters.")
            sigma_sqrd = self.hyperparameters["sigma_sqrd"]
            lam = self.hyperparameters["lambda"]
            b_sqrd = lam / sigma_sqrd if sigma_sqrd != 0 else 1
        else:
            raise ValueError("Un-supported variant was specified. ")
        
        # Call the helper
        self.learned_parameters = self._map_estimation(X, y, sigma_sqrd, b_sqrd)
        self.parameters_assigned = True

    # Helper function: MAP Estimation
    def _map_estimation(self, X, y, sigma_sqrd, b_sqrd):
        print(f"Training using tthe MAP Estimation (σ²={sigma_sqrd}, b²={b_sqrd})")
        X_T = X.T
        XTX = X_T @ X
        
        identity_matrix = np.eye(XTX.shape[0])
        M = XTX + (sigma_sqrd / b_sqrd) * identity_matrix
        
      
        if np.linalg.det(M) == 0:
            raise ValueError("Matrix M is singular !")

        M_inv = np.linalg.inv(M)
        theta = M_inv @ X_T @ y
        
        return theta

    # Prediction function
    def predict(self, X):
        # Sanity check 
        if not self.parameters_assigned:
            raise ValueError("Model parameters have not been learned yet.")
        
        # Compute predictions: z = X * θ
        z = X @ self.learned_parameters
        return z

# Example usage:
if __name__ == "__main__":
    # Training data (X: inputs, y: outputs)
    X_train = np.array([[1, 2], [2, 3], [3, 4]])
    y_train = np.array([5, 7, 9])


    hyperparams = {"variant": "MLE"}


    lin_reg = LinearRegression(hyperparameters=hyperparams)
    lin_reg.train(X_train, y_train)

    # Make predictions
    X_test = np.array([[4, 5], [6, 7]])
    predictions = lin_reg.predict(X_test)
    print("Predictions are:", predictions)
