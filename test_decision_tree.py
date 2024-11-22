import numpy as np
from DecisionTree import DecisionTree
from DecisionTree import RandomForest


# ----------------- Test Dataset -----------------
# This is like a dataset for playing tennis 
# Columns represent features, rows are data points.
X_test = np.array([
    [0, 0, 0, 0],  # Sunny, Hot, High, Weak
    [0, 0, 0, 1],  # Sunny, Hot, High, Strong
    [1, 0, 0, 0],  # Overcast, Hot, High, Weak
    [2, 1, 0, 0],  # Rain, Mild, High, Weak
    [2, 2, 1, 0],  # Rain, Cool, Normal, Weak
    [2, 2, 1, 1],  # Rain, Cool, Normal, Strong
    [1, 2, 1, 1],  # Overcast, Cool, Normal, Strong
    [0, 1, 0, 0],  # Sunny, Mild, High, Weak
    [0, 2, 1, 0],  # Sunny, Cool, Normal, Weak
    [2, 1, 1, 0],  # Rain, Mild, Normal, Weak
    [0, 1, 1, 1],  # Sunny, Mild, Normal, Strong
    [1, 1, 0, 1],  # Overcast, Mild, High, Strong
    [1, 0, 1, 0],  # Overcast, Hot, Normal, Weak
    [2, 1, 0, 1],  # Rain, Mild, High, Strong
])

# Corresponding labels for the dataset above.
y_test = np.array([
    0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0
])



# ---------- Decision Tree Test --
# Creating a decision tree to classify the dataset
decision_tree = DecisionTree(criterion="entropy")  #  entropy to measure split

# Train the tree with the test data (yea I know it's not ideal).
decision_tree.train(X_test, y_test)

# Make predictions using the trained tree
predictions = decision_tree.predict(X_test)
print("Predictions:", predictions)  # Should match labels if overfitting?



# ------ Random Forest Test -----
# Random forest: an ensemble of trees for better results
random_forest = RandomForest(n_trees=5, criterion="entropy")  # Five trees in the forest

# Train the forest using the same data (again, not ideal)
random_forest.train(X_test, y_test)

# Get predictions from the forest
rf_predictions = random_forest.predict(X_test)
print("Random Forest Predictions:", rf_predictions)  # Likely more stable
