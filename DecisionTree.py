import numpy as py
import pandas as pd
from Utilities import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionTree:

    def build_all_tree(splits, features):
        classifiers = [None] * len(features)
        classifiers_accuracy = [0] * len(features)
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splits
        minimum_splits = 2
        maximum_depth = 5
        for i in range(len(features)):
            if i % 1000 == 0: print (f'starting tree {i}')
            temp_X_train = X_train[:, i].reshape(-1, 1)
            classifier = DecisionTreeClassifier(minimum_splits, maximum_depth)
            classifier.fit(temp_X_train, Y_train)

            classifiers[i] = classifier

            temp_X_test = X_test[:, i].reshape(-1, 1)
            Y_pred = classifier.predict(temp_X_test)
            classifiers_accuracy[i] = accuracy_score(Y_test, Y_pred)
        return classifiers, classifiers_accuracy

    def split_data(features, csv_name, labels):
        data = DecisionTree.get_data(features, csv_name)
        labels_df = pd.DataFrame({'Label' : labels})
        data = pd.concat([data, labels_df], axis=1)

        X = data.iloc[:, :-1].values 
        Y = data.iloc[:, -1].values.reshape(-1, 1)

        X_temp, X_train, Y_temp, Y_train = train_test_split(X, Y, test_size=0.3, random_state=42)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

        # print(type(X_train))
        splits = [X_train, Y_train, X_test, Y_test, X_valid, Y_valid]
        return splits

    def get_data(features, csv_name):
        col_names = []
        for i in range(len(features)):
            temp_column_name = f'win_1_feature_{i}'
            col_names.append(temp_column_name)
        return Utilities.read_csv(csv_name, col_names)
    
    def get_partial_data(csv_name, feature_num):
        col_name = f'win_1_feature_{feature_num}'
        return Utilities.read_partial_csv(csv_name, col_name)

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        self.value = value

class DecisionTreeClassifier():
    def __init__(self, minimum_splits = 2, maximum_depth = 2):
        self.root = None

        # stopping condition
        self.minimum_splits = minimum_splits
        self.maximum_depth = maximum_depth
    
    def build_tree(self, training_dataset, current_depth = 0):

        X, Y = training_dataset[:,:-1], training_dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # split until conditons are met
        if num_samples >= self.minimum_splits and current_depth <= self.maximum_depth:
            # find best split
            best_split = self.get_best_split(training_dataset, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], current_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], current_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])
            
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_features):
        # dictionary to save data
        best_split = {
        "info_gain": -float("inf")  # Initialize info_gain to a very small value
        } #btw ini aneh tapi buat feature 19 (dan mungkin lebih banyak lagi), karna suatu alesan dia gak pass len(dataset_left) > 0 and len(dataset_right) > 0
        # atau current_info_gain > max_info_gain. Jadinya gak init "info_gain" dan bikin error di build_tree. Ini bypass doang dan mungkin featurenya bakal gak bisa kepake
        # tapi liat aja nanti, ada 520000 feature, kalo emg ada berapa ratus feature ilang gak bisa kepake harusnya masih bisa.
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            potential_thresholds = np.unique(feature_values)

            for threshold in potential_thresholds:
                # get curent split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if child not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, rigth_y = dataset[:,-1], dataset_left[:,-1], dataset_right[:, -1]
                    # compute information gain
                    current_info_gain = self.information_gain(y, left_y, rigth_y, "gini")
                    # update best split if needed
                    if current_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = current_info_gain
                        max_info_gain = current_info_gain

        return best_split

    def split(self, dataset, feature_index, threshold):
        # fuction to split data 
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_rigth = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_rigth
    
    def information_gain(self, parent, left_child, right_child, mode="entropy"):
        weight_left = len(left_child) / len(parent)
        weight_rigth = len(right_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_left * self.gini_index(left_child) + weight_rigth * self.gini_index(right_child))
        else:
            gain = self.entropy(parent) - (weight_left * self.entropy(left_child) + weight_rigth * self.entropy(right_child))
        return gain
    
    def entropy(self, y):
        # fuction to count entropy
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        # function to count gini index (lebih cepet aja karna gak pake log)
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, Y):

        Y = list(Y)
        return max(Y, key = Y.count)

    def print_tree(self, tree=None, indent=" "):
        # fuction just to print the tree

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        # fuction to train tree 
        dataset = np.concatenate((X, Y), axis = 1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        # fuction to predict new dataset 
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        # fuction to detect single datapoint
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold: return self.make_prediction(x, tree.left)
        else: return self.make_prediction(x, tree.right)

