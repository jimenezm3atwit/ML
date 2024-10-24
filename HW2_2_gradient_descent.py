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


# GradientDescent class that inherits from MLAlgorithmTemplate
class GradientDescent(MLAlgorithmTemplate):
    def __init__(self, hyperparameters=None):
        super().__init__(hyperparameters)

    def gradient_descent(self, X, y):
        # Ensure hyperparameters are provided
        if not self.hyperparameters:
            raise ValueError("Hyperparameters are required to do a gradient descent.")
        
        # Extract hyperparameters
        learn_pace = self.hyperparameters.get("learn_pace", 0.01)
        convergence_threshold = self.hyperparameters.get("convergence_threshold", 1e-6)
        max_iterations = self.hyperparameters.get("max_iterations", 1000)

      
        intercept_column = np.ones((X.shape[0], 1))
        X_augmented = np.hstack((intercept_column, X))

      
        self.learned_parameters = np.zeros(X_augmented.shape[1])

        # Gradient Descent Loop
        for iteration in range(max_iterations):
        
            z = X_augmented @ self.learned_parameters
            probabilities = 1 / (1 + np.exp(-z))

            
            errors = y - probabilities
            gradient = -X_augmented.T @ errors / len(y)

            # Update the parameters using the gradient
            self.learned_parameters -= learn_pace * gradient

         
            if np.linalg.norm(gradient) < convergence_threshold:
                print(f"Convergence reached at iteration {iteration}")
                break

   
        self._update_learned_parameters(self.learned_parameters)

        
        return self.learned_parameters

# Example usage
if __name__ == "__main__":
 
    X_train = np.array([[1, 2], [2, 3], [3, 4]])
    y_train = np.array([0, 1, 0])


    hyperparams = {
        "learn_pace": 0.01,
        "convergence_threshold": 1e-6,
        "max_iterations": 1000
    }


    gradient_descent = GradientDescent(hyperparameters=hyperparams)

    # Perform gradient descent 
    learned_parameters = gradient_descent.gradient_descent(X_train, y_train)

    print("Learned Parameters:", learned_parameters)
