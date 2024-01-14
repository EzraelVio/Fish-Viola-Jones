import numpy as np
from Boosting import *
from DecisionTree import *
from Utilities import *
from sklearn.metrics import accuracy_score

class Cascade:

    def __init__(self):
        self.stages = []

    # fill cascade with stages until there are no more classifier left
    def fill_cascade(self, features, trees, alpha_list, splits):
        print(f'starting to fill cascade...')
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splits
        used_features = 0
        print(f'number of used_features: {used_features}')
        while True:
            if used_features >= len(features): break
            new_cascade = CascadeStage()
            new_cascade.train_stage(features, trees, alpha_list, X_valid, Y_valid, used_features)
            used_features += len(new_cascade.trees) #check the total number of features used
            self.stages.append(new_cascade)
            print(f'finished filling stage: {len(self.stages)}')
        print(f'cascade is finished!')

    # final ultimate strongest cascade classifier! Used in the actual detection to scan sub-windows
    # x_offset and y_offset is the location of the sliding window
    def final_cascade_classification(self, image, x_offset, y_offset):
        scoreboard = [0, 0, 0, 0]
        for i in range(len(self.stages)):
            stage_scoreboard = [0, 0, 0, 0]
            stage_scoreboard = self.stages[i].stage_prediction(image, x_offset, y_offset, stage_scoreboard)

            # check whether the stage return false or a class. If a class then continue
            if stage_scoreboard.index(max(stage_scoreboard)) == 0: break
            else: scoreboard = [scoreboard + stage_scoreboard for scoreboard, stage_scoreboard in zip(scoreboard, stage_scoreboard)]

        return scoreboard.index(max(scoreboard))
    
    def save_to_pickle(self, pickle_name):
        print(f'saving to Pickle...')
        Utilities.dump_to_pickle(f'{pickle_name}', self)
        print(f'complete!')
            
    
class CascadeStage:

    def __init__(self):
        self.features = []
        self.trees = []
        self.alpha_list = []

    def train_stage(self, features, trees, alpha_list, X_valid, Y_valid, used_features):
        detection_rate = 0
        while detection_rate < 0.5:
            if used_features >= len(features): break
            # append weak classifier into stage one by one
            self.features.append(features[used_features])
            self.trees.append(trees[used_features])
            self.alpha_list.append(alpha_list[used_features])

            orderlist = np.arange(len(self.trees))
            validation_prediction = Boosting.strong_prediction(self.trees, orderlist, X_valid, self.alpha_list)
            detection_rate = accuracy_score(Y_valid, validation_prediction)
            used_features += 1
        print(f'features used in this stage: {used_features}')
        
    def stage_prediction(self, image, x_offset, y_offset, scoreboard):
        for i in range(len(self.features)):
            feature_type, x, y, width, height = self.features[i]
            x += x_offset
            y += y_offset
            updated_feature = (feature_type, x, y, width, height)
            feature_value = compute_feature_with_matrix(image, updated_feature)

            data_features = [[feature_value]]

            prediction = self.trees[i].predict(data_features)
            prediction = prediction[0]
            scoreboard[prediction] += 1 * self.alpha_list[i]

        return scoreboard
