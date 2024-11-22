import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

best_model = Ridge(alpha=1.0)
best_model.fit(X, y)
predicted_prices = best_model.predict(X)

# Create new feature: willing to purchase
willing_to_purchase = (predicted_prices >= y).astype(int)
housing_data['WillingToPurchase'] = willing_to_purchase

# Print the number of 1s and 0s in WillingToPurchase
num_yes = sum(willing_to_purchase)
num_no = len(willing_to_purchase) - num_yes
print(f"Number of 'willing to purchase' = 1: {num_yes}")
print(f"Number of 'willing to purchase' = 0: {num_no}")

X_log = housing_data.drop(columns=['MedHouseVal', 'WillingToPurchase'])
y_log = housing_data['WillingToPurchase']

# Train logistic regression model
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

    cm_percentage = cm / cm.sum() * 100
    print(f"Confusion Matrix for training size {percentage * 100}%:")
    print(cm)
    print("Confusion Matrix as Percentage:")
    print(cm_percentage)
    print()

# Plot the confusion matrix metrics as percentages
metrics = ['True Positives', 'False Negatives', 'False Positives', 'True Negatives']
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.ravel()

for idx, metric in enumerate(metrics):
    values = []
    for cm in confusion_matrices:
        if metric == 'True Positives':
            values.append(cm[1, 1])
        elif metric == 'False Negatives':
            values.append(cm[1, 0])
        elif metric == 'False Positives':
            values.append(cm[0, 1])
        elif metric == 'True Negatives':
            values.append(cm[0, 0])
    axs[idx].plot(training_percentages, values, marker='o')
    axs[idx].set_title(f'{metric} vs Training Dataset Size')
    axs[idx].set_xlabel('Training Dataset Size (Percentage)')
    axs[idx].set_ylabel(f'Count of {metric}')

plt.tight_layout()
plt.show()
