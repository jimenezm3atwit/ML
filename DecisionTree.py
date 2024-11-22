import numpy as np
from collections import Counter  # Needed for counting stuff


# -----#---------- Tree Node Class -----

class DecisionTreeNode:
    
    """
    Represents a single node in the decision tree. 
    """
    
    def __init__(self, is_leaf=False, feature=None, label=None):
        # A decision tree node. Could be a split or a decision.
        self.is_leaf = is_leaf  # True if this node makes a decision
        self.feature = feature  # Index of feature to split on
        self.label = label      # The label to assign if it's a leaf
        self.children = {}      # For the split children, if any



# ------------------ Decision Tree Cla------
class DecisionTree:
    
    """
    Decision Tree Classifier. Splits data based on some rules.
    """
    
    def __init__(self, criterion="entropy"):
        # Initialization stuff
        self.criterion = criterion  # "entropy" or "gini"
        self.root = None  # Where the tree starts

    
    def _entropy(self, label_array):
        """
        Calculates entropy. Lower entropy means better splits.
        """
        from math import log2

        label_counts = Counter(label_array)
        total_count = len(label_array)
        entropy_value = 0

        for count in label_counts.values():
            probability = count / total_count
            entropy_value -= probability * log2(probability) if probability > 0 else 0

        return entropy_value

    
    def _gini(self, label_array):
        # Computes gini. This is another way of measuring splits.
        label_counts = Counter(label_array)
        total_count = len(label_array)
        gini_value = 1

        for count in label_counts.values():
            probability = count / total_count
            gini_value -= probability ** 2

        return gini_value

    
    def _best_split(self, X_data, Y_labels):
        """
        Finds the best feature to split on by comparing impurity gains.
        """
        sample_count, feature_count = X_data.shape
        best_feature = None
        best_gain = -float("inf")  # Start with a very low gain
        current_impurity = self._entropy(Y_labels) if self.criterion == "entropy" else self._gini(Y_labels)

        for feature in range(feature_count):
            unique_values = set(X_data[:, feature])  # Get unique feature values
            for value in unique_values:
                left_indices = X_data[:, feature] <= value
                right_indices = ~left_indices

                # Calculate impurity for splits
                left_impurity = (
                    self._entropy(Y_labels[left_indices])
                    if self.criterion == "entropy"
                    else self._gini(Y_labels[left_indices])
                )
                right_impurity = (
                    self._entropy(Y_labels[right_indices])
                    if self.criterion == "entropy"
                    else self._gini(Y_labels[right_indices])
                )

                # Weighted impurity reduction
                left_weight = len(Y_labels[left_indices]) / sample_count
                right_weight = len(Y_labels[right_indices]) / sample_count
                impurity_gain = current_impurity - (
                    left_weight * left_impurity + right_weight * right_impurity
                )

                # Update if a better split is found
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = feature

        return best_feature

    
    def _build_tree(self, feature_matrix, labels):
        # Recursively build the decision tree
        if len(set(labels)) == 1:  # If all labels are the same
            return DecisionTreeNode(is_leaf=True, label=labels[0])

        if feature_matrix.shape[1] == 0:  # No features left to split
            majority_label = Counter(labels).most_common(1)[0][0]
            return DecisionTreeNode(is_leaf=True, label=majority_label)

        # Find the best feature to split on
        best_feature = self._best_split(feature_matrix, labels)
        if best_feature is None:  # If no valid split found
            majority_label = Counter(labels).most_common(1)[0][0]
            return DecisionTreeNode(is_leaf=True, label=majority_label)

        # Make a node for this split
        node = DecisionTreeNode(feature=best_feature)
        node.label = Counter(labels).most_common(1)[0][0]  # Assign majority label for fallback
        unique_values = set(feature_matrix[:, best_feature])

        for value in unique_values:
            indices = feature_matrix[:, best_feature] == value
            child_node = self._build_tree(feature_matrix[indices], labels[indices])
            node.children[value] = child_node

        return node

    
    def train(self, feature_matrix, labels):
        # Train the decision tree using the provided data
        self.root = self._build_tree(feature_matrix, labels)

    
    def predict(self, feature_matrix):
        """
        Predict labels for input data.
        """
        def traverse(node, row):
            if node.is_leaf:
                return node.label
            feature_value = row[node.feature]
            child = node.children.get(feature_value)
            if child is None:  # If feature value unseen
                return node.label if node.label is not None else 0  # Default to 0
            return traverse(child, row)

        return np.array([traverse(self.root, row) for row in feature_matrix], dtype=int)



# ---------- Random Forst Class --

class RandomForest:
    
    """
    Random Forest: A collection of decision trees.
    """
    
    def __init__(self, n_trees=10, criterion="entropy"):
        self.n_trees = n_trees  # How many trees to use
        self.criterion = criterion
        self.trees = []  # Where all trees are stored

    
    def _bootstrap_sample(self, feature_matrix, labels):
        # Make a random sample of the data with replacement
        sample_count = feature_matrix.shape[0]
        indices = np.random.choice(sample_count, sample_count, replace=True)
        return feature_matrix[indices], labels[indices]

    
    def train(self, feature_matrix, labels):
        """
        Train the random forest using bootstrap samples.
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(criterion=self.criterion)
            X_sample, Y_sample = self._bootstrap_sample(feature_matrix, labels)
            tree.train(X_sample, Y_sample)
            tree.root.label = Counter(Y_sample).most_common(1)[0][0]  # Assign fallback label
            self.trees.append(tree)

    
    def predict(self, feature_matrix):
        """
        Predict by voting on all tree outputs.
        """
        tree_predictions = [tree.predict(feature_matrix) for tree in self.trees]
        tree_predictions = np.array(tree_predictions, dtype=int)
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
        return majority_votes
