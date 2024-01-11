import numpy as np
from DecisionTree import *
from Utilities import *
from sklearn.metrics import accuracy_score

class Boosting:

    def training_strong_classifier(features, trees, splits, accuracy, pickle_name):
        X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splits

        orderlist = np.arange(len(accuracy))
        orderlist = Boosting.get_initial_sorted_accuracy(accuracy, orderlist)
        image_weights = Boosting.initialize_weight(Y_test)

        initial_accuracy = float('-inf')
        current_accuracy = 0
        iteration = 0
        limit = 100

        # start boosting loop. Will stop when accuracy fell or iteration hit limit
        while current_accuracy > initial_accuracy or iteration > limit:
            alpha_list = Boosting.start_boosting(trees, X_test, Y_test, image_weights, orderlist)
            validation_prediction = Boosting.strong_prediction(trees, orderlist, X_valid, alpha_list)

            initial_accuracy = current_accuracy
            current_accuracy = accuracy_score(Y_valid, validation_prediction)
            # print(Y_valid)
            # print(validation_prediction)
            print(f'current accuracy: {current_accuracy}')
            orderlist = Boosting.get_sorted_accuracy(alpha_list, orderlist)
            iteration += 1

        # saving trees, related features and its order in pickle
        final_trees = np.empty(len(orderlist), dtype=object)
        final_features = np.empty(len(orderlist), dtype=object)
        for i in range(len(orderlist)):
            final_trees[i] = trees[orderlist[i]]
            final_features[i] = features[orderlist[i]]

        pickle_this = PickleTreeFinal(final_features, final_trees, alpha_list)
        Utilities.dump_to_pickle(f'{pickle_name}', pickle_this)
        


    def start_boosting(trees, X_test, Y_test, image_weights, orderlist):
        alpha_list = np.zeros(len(orderlist))
        for i in range(len(orderlist)):
            # make prediction with i-th tree
            treeN = orderlist[i]
            prediction  = trees[treeN].predict(X_test)

            # calculate error of the tree
            indicator = np.array(np.array(prediction).astype(int) != Y_test.flatten(), dtype = float)
            epsilon = np.sum(image_weights * indicator) / np.sum(image_weights)

            # calculate the weight of the tree
            alpha = 0.5 * np.log((1 - epsilon) / (epsilon + 1e-10)) #1e-10 const added to prevent div by 0
            alpha_list[i] = alpha

            # update the weight for the samples
            image_weights *= np.exp(alpha * indicator)
            image_weights /= np.sum(image_weights) #normalize weight so it is closer to 1

        return alpha_list
    
    def strong_prediction(trees, orderlist, X_valid, alpha_list):
        predictions = [0] * len(X_valid)
        scoreboard = [[0, 0, 0, 0] for _ in range(len(X_valid))]
        for i in range(len(orderlist)):
            tree_index = orderlist[i]
            prediction = trees[tree_index].predict(X_valid)

            for j in range(len(prediction)):
                weak_learner_prediction = int(prediction[j])
                scoreboard[j][weak_learner_prediction] += 1 * alpha_list[i]
        
        for k in range(len(prediction)):
                predictions[k] = scoreboard[k].index(max(scoreboard[k]))
        return predictions



    # get initial order of boosting, create orderlist = np.arange(len(accuracy)) 1st
    # accuracy can be either from SKlearn comparison or alpha_list
    def get_initial_sorted_accuracy(accuracy, orderlist):
        print(f'initial feature count: {np.shape(orderlist)}')
        accuracy_threshold = 0.5 # change 0.5 to whatever needed. 0.5 seems logical enough considering the random guessing concept
        accuracy, orderlist = zip(*sorted(zip(accuracy, orderlist), reverse = True))
        orderlist = [classifier for accuracy, classifier in zip(accuracy, orderlist) if accuracy >= accuracy_threshold]
        print(f'after elimination feature count: {np.shape(orderlist)}')
        print(f'first then features: {orderlist[:10]}')
        return orderlist
    
    # accuracy test without feature elimination, used in Boosting loop
    def get_sorted_accuracy(accuracy, orderlist):
        accuracy, orderlist = zip(*sorted(zip(accuracy, orderlist), reverse = True))
        return orderlist

    # just assign weight according to numbers of image used. Use Y_test as input
    # ini harus diubah ntar sih, karena berantakan aja kalo disini dan gak jelas
    def initialize_weight(test_images):
        image_weights = np.ones(len(test_images)) / len(test_images)
        return image_weights
    
