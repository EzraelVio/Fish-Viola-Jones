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
        # initialize separate Dataframe for features list (Data gembrot)
        temp_window_value1 = np.zeros(len(images), dtype=object)
        temp_window_value2 = np.zeros(len(images), dtype=object)
        temp_window_value3 = np.zeros(len(images), dtype=object)
        image_ids = np.arange(len(images))

        for i in range(len(images)):
            new_Data = Dataset(images[i], labels[i], features)
            temp_window_value1[i] = new_Data.window_1_features

        window_1_feature = {
            'image_ids' : image_ids,
        }
        for i in range(len(features)):
            temp_value = np.zeros(len(images))
            for j in range(len(images)):
                temp_value[j] = temp_window_value1[j][i]
            column_name = f'win_1_feature_{i}'
            window_1_feature[column_name] = temp_value
        
        directory = "Data/" + csv_name + "_window_0.csv"

        window_1_feature = pd.DataFrame(window_1_feature)
        window_1_feature.to_csv(directory, index=False)

        for i in range(len(images)):
            new_Data = Dataset(images[i], labels[i], features)
            temp_window_value2[i] = new_Data.window_2_features

        window_2_feature = {
            'image_ids' : image_ids,
        }
        for i in range(len(features)):
            for j in range(len(images)):
                temp_value[j] = temp_window_value2[j][i]
            column_name = f'win_2_feature_{i}'
            window_2_feature[column_name] = temp_value
        
        directory = "Data/" + csv_name + "_window_1.csv"

        window_2_feature = pd.DataFrame(window_2_feature)
        window_2_feature.to_csv(directory, index=False)

        for i in range(len(images)):
            new_Data = Dataset(images[i], labels[i], features)
            temp_window_value3[i] = new_Data.window_3_features

        window_3_feature = {
            'image_ids' : image_ids,
        }
        for i in range(len(features)):
            for j in range(len(images)):
                temp_value[j] = temp_window_value3[j][i]
            column_name = f'win_1_feature_{i}'
            window_3_feature[column_name] = temp_value
        
        directory = "Data/" + csv_name + "_window_2.csv"

        window_3_feature = pd.DataFrame(window_3_feature)
        window_3_feature.to_csv(directory, index=False)
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