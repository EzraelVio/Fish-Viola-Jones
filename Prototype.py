import numpy as np
import os
import cv2
from Cascade import *

class prototype:
    def get_label(directory):
        # add more to add more class
        if directory == "fish_dataset\\cascade_train\\abudefduf": return 1
        if directory == "fish_dataset\\cascade_train\\amphiprion": return 2
        if directory == "fish_dataset\\cascade_train\\chaetodon": return 3
        else: return 0

    def load_images(directory):
        images=[]
        labels=[]

        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # yg dikomen ini buat daun 700x524. Ikan pake 350x200
                image = cv2.resize(image, (350, 200))
                # image = cv2.resize(image, (700, 524))
                images.append(image)
                labels.append(prototype.get_label(directory))
        return np.array(images), np.array(labels)

    def start_load():
        abudefduf_images, abudefduf_labels = prototype.load_images("fish_dataset\\cascade_train\\abudefduf")
        amphiprion_images, amphiprion_labels = prototype.load_images("fish_dataset\\cascade_train\\amphiprion")
        chaetodon_images, chaetodon_labels = prototype.load_images("fish_dataset\\cascade_train\\chaetodon")

        images = np.concatenate((abudefduf_images, amphiprion_images, chaetodon_images), axis = 0)
        labels = np.concatenate((abudefduf_labels, amphiprion_labels, chaetodon_labels), axis = 0)

        return images, labels
    


    def find_cascade_alpha(images, labels, cascade, index_of_location):
        image_weights = prototype.initialize_weight(images)
        orderlist = np.arange(len(index_of_location))

        initial_accuracy = float('-inf')
        current_accuracy = 0
        iteration = 0
        limit = 100 #change according to needs

        # start boosting loop. Will stop when accuracy fell or iteration hit limit
        while True:
            alpha_list = prototype.start_boosting_index(images, labels, cascade, index_of_location, image_weights)
            alpha_list = Boosting.start_boosting(trees, X_test, Y_test, image_weights, orderlist)
            validation_prediction = Boosting.strong_prediction(trees, orderlist, X_valid, alpha_list)

            initial_accuracy = current_accuracy
            print(f'current initial accuracy: {initial_accuracy}')
            current_accuracy = accuracy_score(Y_valid, validation_prediction)
            print(f'current after boosting accuracy: {current_accuracy}')
            
            # check whether accruacy deteriorate or limit hit
            if current_accuracy <= initial_accuracy or iteration >= limit:
                print('final accuracy deteriorate, rolling back to last iteration...')
                alpha_list = last_iteration_alpha_list
                orderlist = last_iteration_orderlist
                break
            
            print('starting over. Saving alpha...')
            alpha_list, orderlist = Boosting.get_sorted_accuracy(alpha_list, orderlist)
            last_iteration_alpha_list = alpha_list
            last_iteration_orderlist = orderlist
            iteration += 1


        # saving trees, related features and its order in pickle
        final_trees = np.empty(len(orderlist), dtype=object)
        final_features = np.empty(len(orderlist), dtype=object)
        for i in range(len(orderlist)):
            final_trees[i] = trees[orderlist[i]]
            final_features[i] = features[orderlist[i]]

        pickle_this = PickleTreeFinal(final_features, final_trees, alpha_list)
        Utilities.dump_to_pickle(f'{pickle_name}', pickle_this)
        


    def start_boosting_index(images, labels, cascade, index_of_location, image_weights):
        print('Boosting index...')
        alpha_list = np.zeros(len(index_of_location))
        for i in range(len(index_of_location)):
            prediction = np.zeros(len(images))
            for j in range(images):
                prediction[j] = cascade.final_cascade_classification(images[j], index_of_location[i][1], index_of_location[i][0])

            # calculate error of the tree
            indicator = np.array(prediction != labels, dtype=float)
            epsilon = np.sum(image_weights * indicator) / np.sum(image_weights)

            # calculate the weight of the tree
            alpha = 0.5 * np.log((1 - epsilon) / (epsilon + 1e-10)) + np.log(4 - 1) #1e-10 const added to prevent div by 0. 4 is number of class
            if alpha < 1e-10: alpha = 1e-10 #1e-10 const added to prevent alpha getting too small in np.exp(alpha * indicator) later
            alpha_list[i] = alpha

            # update the weight for the samples so the sum of image_weight will be close to 1 for the next iteration
            image_weights *= np.exp(alpha * indicator)
            image_weights /= np.sum(image_weights)

        return alpha_list
    
    def list_of_cascade_predict(cascade, index_of_location, images, alpha_list):
        predictions = [0] * len(images)
        scoreboard = [[0, 0, 0, 0] for _ in range(len(images))]

        for i in range()

        for i in range(len(orderlist)):
            tree_index = orderlist[i]
            prediction = trees[tree_index].predict(X_valid)

            # add score to scoreboard according to results and alpha value of tree
            for j in range(len(prediction)):
                weak_learner_prediction = int(prediction[j])
                scoreboard[j][weak_learner_prediction] += 1 * alpha_list[i]
        
        # return score to the main scoreboard
        for k in range(len(prediction)):
                # print(f'scoreboard {k}: {scoreboard[k]}')
                predictions[k] = scoreboard[k].index(max(scoreboard[k]))
        return predictions
    
    # just assign weight according to numbers of image used. Use Y_test as input
    def initialize_weight(test_images):
        image_weights = np.ones(len(test_images)) / len(test_images)
        return image_weights