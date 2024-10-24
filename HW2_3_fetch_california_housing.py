import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load the California Housing Dataset from scikit-learn
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame

# Inspect the dataset to understand its structure
print(df.head())

# Step 2: Training Multiple Linear Regression Models
# Splitting the dataset into features (X) and target (y)
X = df.drop(columns=['MedHouseVal'])  # 'MedHouseVal' is the target
y = df['MedHouseVal']

# Function to train and evaluate multiple linear regression models
def train_and_evaluate_models(X, y, train_sizes):
    rmse_values = []
    models = []

    for train_size in train_sizes:
        # Split the data into training and testing sets based on the specified train size
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

        # Train a linear regression model using numpy's least squares solution
        # Adding intercept term to features
        X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        # Calculate weights using Normal Equation: theta = (X^T X)^-1 X^T y
        theta = np.linalg.inv(X_train_augmented.T @ X_train_augmented) @ X_train_augmented.T @ y_train

        # Store the model parameters (weights)
        models.append(theta)

        # Make predictions on the test set
        y_pred = X_test_augmented @ theta

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_values.append(rmse)

    return rmse_values, models

# Define the different training sizes to experiment with
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8]

# Train models and get RMSE values
rmse_values, models = train_and_evaluate_models(X.values, y.values, train_sizes)

# Step 3: Plotting RMSE vs Training Dataset Size
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, rmse_values, marker='o')
plt.xlabel('Training Dataset Size (Percentage)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE vs Training Dataset Size for California Housing Dataset')
plt.grid()
plt.show()

# Step 4: Feature Analysis
# Select the best model (with lowest RMSE, but not overfitted)
best_model_index = np.argmin(rmse_values)
best_theta = models[best_model_index]

# Analyze the learned parameters
feature_names = ['Intercept'] + list(X.columns)
for name, coef in zip(feature_names, best_theta):
    impact = 'positive' if coef > 0 else 'negative' if coef < 0 else 'no impact'
    print(f"Feature '{name}' has a {impact} impact on the housing price, with coefficient: {coef}")

# Determining the most impactful feature
most_impactful_feature_index = np.argmax(np.abs(best_theta[1:]))  # Ignore the intercept
most_impactful_feature = feature_names[most_impactful_feature_index + 1]
print(f"\nThe most impactful feature on the housing price is: {most_impactful_feature}")

