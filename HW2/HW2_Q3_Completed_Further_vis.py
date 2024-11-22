import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import fetch_california_housing
import math

# Load the California housing dataset
california_housing = fetch_california_housing(as_frame=True)
housing_data = california_housing.frame

# Feature matrix X and target vector y 
X = housing_data.drop(columns=['MedHouseVal'])
y = housing_data['MedHouseVal']

# Define training percentages and regression models
training_percentages = [0.1, 0.2, 0.4, 0.6, 0.8]
linear_models = [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.1)]
model_names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
rmse_results = {name: [] for name in model_names}

# Train linear regression models and calculate RMSE
for percentage in training_percentages:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=percentage, random_state=42)

    for model, name in zip(linear_models, model_names):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        rmse_results[name].append(rmse)

# Plot RMSE vs Training Dataset Size
plt.figure(figsize=(10, 6))
for name in model_names:
    plt.plot(training_percentages, rmse_results[name], marker='o', label=name)
plt.xlabel('Training Dataset Size (Percentage)')
plt.ylabel('RMSE')
plt.title('RMSE vs Training Dataset Size for Linear Regression Models')
plt.legend()
plt.show()

# Use the best model (Ridge Regression) for classification analysis
best_model = Ridge(alpha=1.0)
best_model.fit(X, y)
predicted_prices = best_model.predict(X)

# Create a new feature: "willing to purchase" (1 if predicted >= actual, otherwise 0)
willing_to_purchase = (predicted_prices >= y).astype(int)
housing_data['WillingToPurchase'] = willing_to_purchase

# Print the number of 'willing to purchase' = 1 and 0
num_yes = sum(willing_to_purchase)
num_no = len(willing_to_purchase) - num_yes
print(f"Number of 'willing to purchase' = 1: {num_yes}")
print(f"Number of 'willing to purchase' = 0: {num_no}")

# Prepare data for logistic regression
X_log = housing_data.drop(columns=['MedHouseVal', 'WillingToPurchase'])
y_log = housing_data['WillingToPurchase']

# Train logistic regression model and collect confusion matrices
confusion_matrices = []
for percentage in training_percentages:
    X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, train_size=percentage, random_state=42)

    # Train logistic regression model
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = log_reg.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

# Plot confusion matrices side by side
fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(20, 5))

for i, (cm, percentage) in enumerate(zip(confusion_matrices, training_percentages)):
    ax = axes[i]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax, cbar=False, annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(f"Train Size: {int(percentage * 100)}%", fontsize=12, weight="bold")
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xticklabels(['No', 'Yes'], fontsize=10)
    ax.set_yticklabels(['No', 'Yes'], fontsize=10)

plt.tight_layout()
plt.show()
