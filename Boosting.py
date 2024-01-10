import numpy as np
from DecisionTree import *
from sklearn.metrics import accuracy_score

class Boosting:

    def training_strong_classifier(trees, splits, accuracy):
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splits
        orderlist = np.arange(len(accuracy))
        orderlist = Boosting.get_initial_sorted_accuracy(accuracy, orderlist)
        image_weights = Boosting.initialize_weight(Y_test)

        while 
            alpha_list = Boosting.start_boosting(trees, X_test, Y_test, image_weights, orderlist)


    def start_boosting(trees, X_test, Y_test, image_weights, orderlist):
        alpha_list = np.zeros(len(orderlist))
        for i in range(len(orderlist)):
            # make prediction with i-th tree
            treeN = orderlist[i]
            prediction  = trees[treeN].predict(X_test)

            # print(image_weights)

            # calculate error of the tree
            indicator = np.array(np.array(prediction).astype(int) != Y_test.flatten(), dtype = float)
            epsilon = np.sum(image_weights * indicator) / np.sum(image_weights)
            print(image_weights)
            print(indicator)

            # epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)
            
            # calculate the weight of the tree
            alpha = 0.5 * np.log((1 - epsilon) / (epsilon + 1e-10)) #1e-10 const added to prevent div by 0
            alpha_list[i] = alpha
            # alpha = np.clip(alpha, 0, 1)
            print(alpha)
            # print(np.shape(alpha))
            # print(np.shape(indicator))

            # update the weight for the samples
            image_weights *= np.exp(alpha * indicator)
            image_weights /= np.sum(image_weights) #normalize weight so it is closer to 1

        print(sum(alpha_list))
        return alpha_list
    
    def strong_prediction(trees, orderlist, X_valid):
        predictions = None * len(X_valid)
        scoreboard = [0, 0, 0, 0] * len(X_valid)
        for i in range(len(orderlist)):
            prediction = trees[i].predict(X_valid)

            for j in range(len(prediction)):
                scoreboard[j][prediction[j]] = scoreboard[j][prediction[j]] + 1



    # get initial order of boosting, create orderlist = np.arange(len(accuracy)) 1st
    # accuracy can be either from SKlearn comparison or alpha_list
    def get_initial_sorted_accuracy(accuracy, orderlist):
        print(np.shape(orderlist))
        accuracy_threshold = 0.5
        accuracy, orderlist = zip(*sorted(zip(accuracy, orderlist), reverse = True))
        orderlist = [classifier for accuracy, classifier in zip(accuracy, orderlist) if accuracy >= accuracy_threshold]
        print(np.shape(orderlist))
        return orderlist

    # just assign weight according to numbers of image used. Use Y_test as input
    # ini harus diubah ntar sih, karena berantakan aja kalo disini dan gak jelas
    def initialize_weight(test_images):
        image_weights = np.ones(len(test_images)) / len(test_images)
        return image_weights
    
