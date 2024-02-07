from DecisionTree import *
from LoadImages import *
from HaarFeatures import *
from IntegralImage import *
from Dataset import *
from Utilities import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

images, labels = combine_dataset()

# note buat besok:
# ini get data dari datasetnya buat skg make data dari 520k features yg ada
# buat besok benerin supaya dia cuma baca 1 kolom aja, dan juga class dari tiap gambar keknya harus masuk CSV
# artinya CSV harus diremade ato gak harus ada cara buat nge-mend labels + 1 kolom buat bikin tree. Dan proses ini tetep ntar bakal jalan 520k kali

# for testing only
features = generate_features(50, 50)
print(len(features))
print(labels[0])

csv_name = "fish2_window_0"
feature_num = 300000
# data = DecisionTree.get_data(features, csv_name)
data = DecisionTree.get_partial_data(csv_name, feature_num)
# data2 = DecisionTree.get_partial_data(csv_name, feature_num2)
labels_df = pd.DataFrame({'Label' : labels})

data = pd.concat([data, labels_df], axis=1)

print(data)

X = data.iloc[:, :-1].values 
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
print(X_train)

classifier = DecisionTreeClassifier(3, 3)
classifier.fit(X_train, Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test)
print("accuracy:")
print (accuracy_score(Y_test, Y_pred))