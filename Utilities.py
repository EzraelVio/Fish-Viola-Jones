import numpy as np
import pandas as pd
import pickle
from Dataset import *

class PickelData:
    def __init__(self, id, labels, images):
        self.id = id
        self.labels = labels
        self.images = images

class PickleTree:
    def __init__(self, features, trees, accuracies):
        self.feature_num = np.arange(len(features))
        self.trees = trees
        self.accuracies = accuracies

class PickleTreeFinal:
    def __init__(self, features, trees, alpha_list):
        self.features = features
        self.trees = trees
        self.alpha_list = alpha_list

class Utilities:

    def write_csv(images, labels, features, csv_name):
        print("starting write_csv")
        for window_num in range(2, 3):
            temp_window_values = np.zeros((len(images), len(features)), dtype=object)
            image_ids = np.arange(len(images))

            for i in range(len(images)):
                new_data = Dataset(images[i], labels[i], features)
                if window_num == 0:
                    temp_window_values[i] = new_data.window_1_features
                elif window_num == 1:
                    temp_window_values[i] = new_data.window_2_features
                elif window_num == 2:
                    temp_window_values[i] = new_data.window_3_features

            window_feature = {'image_ids': image_ids}
            for i in range(len(features)):
                column_name = f'win_{window_num + 1}_feature_{i}'
                window_feature[column_name] = temp_window_values[:, i]

            directory = f"Data/{csv_name}_window_{window_num}.csv"

            window_feature = pd.DataFrame(window_feature)
            window_feature.to_csv(directory, index=False)
        print("csv write complete!")

    def read_csv(csv_name, col_names):
        # used to read all column
        directory = "Data/" + csv_name + ".csv"
        data = pd.read_csv(directory, skiprows=1, header=None, names = col_names)
        return data
    
    def read_partial_csv(csv_name, col_name):
        # used to read only specified feature colomn to train the decision tree
        directory = "Data/" + csv_name + ".csv"
        data = pd.read_csv(directory, usecols = [col_name])
        return data
    
    def dump_to_pickle(file_name, object):
        directory = "Data/" + file_name + ".pickle"
        with open(directory, 'wb') as file:
            pickle.dump(object, file)

    def read_from_pickle(file_name):
        directory = "Data/" + file_name + ".pickle"
        with open(directory, 'rb') as file:
            return pickle.load(file)