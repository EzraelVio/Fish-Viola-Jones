import numpy as np
from Boosting import *
from DecisionTree import *

class Cascade:

    initial_fpr = 0.5 #fpr = false positive rate
    initial_detection_rate = 0.99
    min_acceptible_dr = 0.95

    # use trees and alpha_list that has been boosted
    def evaluate_cascade(cascade, X_valid, Y_valid, trees, alpha_list):
        number_of_example = len(X_valid)
        false_positives = 0
        detection = 0

        for i in range(number_of_example):
            sample = X_valid.iloc[i]
            label = Y_valid.iloc[i]

            for stage_classifiers in cascade:
                stage_decision = "positive"

                for trees, alpha_list in stage_classifiers:
                    orderlist = np.arange(len(trees))
                    prediction = Boosting.strong_prediction(trees, orderlist,  X_valid, alpha_list)[0]

                    if prediction == 0:
                        stage_decision = "negative"
                        break

                if stage_decision == "negative":
                    break
                else:
                    detection += 1

            if stage_decision == "negative":
                false_positives += 1

        fpr = false_positives / number_of_example
        detection_rate = detection / number_of_example