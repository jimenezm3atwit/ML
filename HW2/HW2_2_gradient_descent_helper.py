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


# GradientDescentHelper class that inherits from MLAlgorithmTemplate
class GradientDescentHelper(MLAlgorithmTemplate):
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

        # Randomly initialize parameters
        num_features = X.shape[1] + 1  # +1 for the intercept term
        self.learned_parameters = np.random.randn(num_features)

 
        intercept_column = np.ones((X.shape[0], 1))
        X_augmented = np.hstack((intercept_column, X))

        # Gradient Descenting loop
        for iteration in range(max_iterations):
  
            theta_prev = self.learned_parameters.copy()

           
            z = X_augmented @ theta_prev
            probabilities = 1 / (1 + np.exp(-z))  
            errors = y - probabilities
            gradient = -X_augmented.T @ errors / len(y)  # Computed the gradient

            # Update parameters using the gradient
            self.learned_parameters -= learn_pace * gradient

            
            diff = self.learned_parameters - theta_prev
            l2_norm_squared = np.dot(diff, diff)  
            if l2_norm_squared <= convergence_threshold:
                print(f"Convergence reached at iteration {iteration}")
                break

        print(f"Gradient descent completed after {iteration+1} iterations")
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

    
    gd_helper = GradientDescentHelper(hyperparameters=hyperparams)

    # Perform gradient descent 
    learned_parameters = gd_helper.gradient_descent(X_train, y_train)

    print("Learned Parameters:", learned_parameters)
