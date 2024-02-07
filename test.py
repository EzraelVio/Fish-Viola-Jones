import pandas as pd
from DecisionTree import *
from Utilities import *
from Boosting import *
from HaarFeatures import *
from LoadImages import *

# images, labels = combine_dataset()
# print(np.shape(images))
# print(labels)

# dec_tree = Utilities.read_from_pickle('window_0_decision_trees')
# print(dec_tree.accuracies)
# Boosting.training_strong_classifier(features, trees, splits, accuracies, pickle_name)

alp = Utilities.read_from_pickle('window_0_strong_classsifier')
count = 0
print(len(alp.trees))
print(alp.features[:20])
print(alp.alpha_list[:20])
# print(alp.accuracies[3738])
alp.trees[0].print_tree()
for i in range(len(alp.trees)):
    if alp.alpha_list[i] > 0.4:
        count +=1
print(count)


# # Assuming you have a CSV file and you read it into a DataFrame
# initial_features = generate_features(50, 50)
# df = DecisionTree.get_data(initial_features, 'fish2_window_0')
# labels_df = pd.DataFrame({'Label' : labels})
# df = pd.concat([df, labels_df], axis=1)

# # Get the number of columns
# num_columns = df.shape[1]

# print(f'The number of features is: {len(initial_features)}')
# print(f'The number of columns is: {num_columns}')
# X = df.iloc[:, 1:-1].values
# num_columns = X.shape[1]
# print(f'The number of features is: {len(initial_features)}')
# print(f'The number of columns is: {num_columns}')
# # print(X[:,:10])
# print(X[:, 100000].reshape(-1, 1))

