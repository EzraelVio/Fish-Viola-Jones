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
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splits
        while True:
            if len(features) == 0: break
            new_cascade = CascadeStage()
            new_cascade.train_stage(features, trees, alpha_list, X_valid, Y_valid)
            self.stages.append(new_cascade)

    # final ultimate strongest cascade classifier! Used in the actual detection to scan sub-windows
    def final_cascade_classification(self, image, x_offset, y_offset):
        scoreboard = np.array([0, 0, 0, 0])
        for i in range(len(self.stages)):
            stage_scoreboard = np.array([0, 0, 0, 0])
            stage_scoreboard = self.stages[i].stage_prediction(image, x_offset, y_offset, stage_scoreboard)

            # check whether the stage return false or a class. If a class then continue
            if stage_scoreboard.index(max(stage_scoreboard)) == 0: break
            else: scoreboard += stage_scoreboard

        return scoreboard.index(max(scoreboard))
    
    def save_to_pickle(self, pickle_name):
        Utilities.dump_to_pickle(f'{pickle_name}', self)
            
    
class CascadeStage:

    def __init__(self):
        self.features = []
        self.trees = []
        self.alpha_list = []

    def train_stage(self, features, trees, alpha_list, X_valid, Y_valid):
        detection_rate = 0
        while detection_rate < 0.5:
            if len(features) == 0: break
            self.features.append(features.pop(0))
            self.trees.append(trees.pop(0))
            self.alpha_list.append(alpha_list.pop(0))
            orderlist = np.arange(len(self.trees))

            validation_prediction = Boosting.strong_prediction(self.trees, orderlist, X_valid, self.alpha_list)
            detection_rate = accuracy_score(Y_valid, validation_prediction)
        

    def stage_prediction(self, image, x_offset, y_offset, scoreboard):
        for i in range(len(self.features)):
            feature_type, x, y, width, height = self.features[i]
            x += x_offset
            y += y_offset
            updated_feature = (feature_type, x, y, width, height)
            data_features = compute_feature_with_matrix(image, 0, updated_feature)

            prediction = self.trees[i].predict(data_features)
            scoreboard[prediction] += 1 * self.alpha_list[i]

        return scoreboard
