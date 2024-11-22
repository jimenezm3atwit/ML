import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from DecisionTree import DecisionTree, RandomForest


# ------- Analyze Dataset ----

def analyze_dataset(X, y, dataset_name):
    """
    Analyze dataset for basic stats like variance and class counts.
    """
    print(f"\n=== {dataset_name} Analysis ===")
    print(f"Class Distribution: {np.bincount(y)}")  # Prints how many samples per class
    print(f"Feature Variance:\n{np.var(X, axis=0)}")  # Variance of features, for fun?


# --------- Balance Dataset -----

def balance_dataset(X, y):
    """
    Balance the dataset using SMOTE to deal with imbalanced classes.
    """
    smote = SMOTE(random_state=42)  # Oversampling to even out classes
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced



# ---- Test Model on Dataset -----

def test_model_on_dataset(X, y, model, model_name):
    """
    Train and evaluate a model on the given dataset.
    """
    # Splitting dataset into train and test (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training the model
    model.train(X_train, y_train)

    # Getting predictions on the test data
    predictions = model.predict(X_test)
    predictions = np.array(predictions, dtype=int)  # Just making sure it's integers

    # Calculating accuracy and showing results
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Classification Report:\n{classification_report(y_test, predictions)}")


# -------- Synthetic Dataset 

# Making a synthetic dataset for testing (500 samples, 10 features)
X_synthetic, y_synthetic = make_classification(
    n_samples=500, n_features=10, n_informative=8, n_classes=2, weights=[0.8, 0.2], random_state=42
)
analyze_dataset(X_synthetic, y_synthetic, "Synthetic Dataset")

# Balancing the synthetic dataset with SMOTE
X_synthetic_balanced, y_synthetic_balanced = balance_dataset(X_synthetic, y_synthetic)
analyze_dataset(X_synthetic_balanced, y_synthetic_balanced, "Balanced Synthetic Dataset")

print("\n=== Testing on Balanced Synthetic Dataset ===")
test_model_on_dataset(X_synthetic_balanced, y_synthetic_balanced, DecisionTree(criterion="entropy"), "DecisionTree (Balanced Synthetic)")
test_model_on_dataset(X_synthetic_balanced, y_synthetic_balanced, RandomForest(n_trees=50, criterion="entropy"), "RandomForest (Balanced Synthetic)")



# ----- Iris Dataset 
# Load the Iris dataset (classic ML dataset)
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
analyze_dataset(X_iris, y_iris, "Iris Dataset")

print("\n=== Testing on Iris Dataset ===")
test_model_on_dataset(X_iris, y_iris, DecisionTree(criterion="entropy"), "DecisionTree (Iris)")
test_model_on_dataset(X_iris, y_iris, RandomForest(n_trees=50, criterion="entropy"), "RandomForest (Iris)")




# test_model_on_dataset(X_synthetic_balanced, y_synthetic_balanced, RandomForest(n_trees=50, criterion="entropy"), "RandomForest (Balanced Synthetic)")  # RandomForest for now








#mote = SMOTE(random_state=42)  # Oversampling to even out classes
   # X_balanced, y_balanced = smote.fit_resample(X, y)
    #return X_balanced, y_balanced
