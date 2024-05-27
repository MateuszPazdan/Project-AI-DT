import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import hickle as hkl
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt 

# Load data from hickle file
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('./badanie_metaparamterow/haberman.hkl')

# Change target values from 1,2 to 0,1
y_t_s -= 1

# Transpose input features and target for row-wise orientation
x = x_n_s.T
y_t = np.squeeze(y_t_s)

# Define metaparameters to explore
criterion_vec = [ "entropy"] 
max_depth_vec = np.array(range(1, 21, 1)) 
splitter_vec = ["best", "random"]
min_samples_split_vec = np.array(range(2, 21, 1))
min_samples_leaf_vec = np.array(range(1, 10, 1))
min_weight_fraction_leaf_vec = np.linspace(0, 0.3, 10)
max_features_vec = np.arange(1, 4, 1)
max_leaf_nodes_vec = np.arange(2, 21, 2)
min_impurity_decrease_vec = np.linspace(0, 0.2, 10)
ccp_alpha_vec = np.linspace(0, 0.2, 10)

# Data and target variables
data = x
target = y_t

# Set the number of folds for cross-validation
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

# Initialize a dictionary to store the accuracy for each combination of metaparameters
results = {}

# Iterate over all combinations of metaparameters
for max_depth in max_depth_vec:
    for splitter in splitter_vec:
        PK_vec = np.zeros(CVN)
        for i, (train, test) in enumerate(skfold.split(data, target), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]

            # Initialize the Decision Tree Classifier with the current metaparameters
            decision_tree = DecisionTreeClassifier(
                random_state=0,
                criterion="entropy",
                max_depth=max_depth,
                splitter=splitter,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.03,
                max_features=3,
                max_leaf_nodes=10,
                min_impurity_decrease=0,
                ccp_alpha=0
            )
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            # Number of test samples
            n_test_samples = test.size
            # Calculate the accuracy for this fold
            PK_vec[i] = np.sum(result == y_test) / n_test_samples

        # Calculate the mean accuracy for this combination of metaparameters
        mean_accuracy = np.mean(PK_vec)
        params = ( max_depth, splitter)
        results[params] = mean_accuracy
        print(f"Params: {params} | Accuracy: {mean_accuracy}")

