import numpy as np
import pandas as pd

from keras.utils import to_categorical
from sklearn import neighbors
from sklearn.model_selection import train_test_split

data_file = 'data/iris.data'
label_name = 'class'
num_classes = 3
test_ratio = 0.1
random_seed = 2

# load CSV
dataset = pd.read_csv(data_file)
dataset.replace('Iris-setosa', 0, inplace=True)
dataset.replace('Iris-versicolor', 0, inplace=True)
dataset.replace('Iris-virginica', 0, inplace=True)

# split features & labels and randomise
features = np.array(dataset.drop([label_name], 1))
labels = np.array(dataset[label_name])

# categorise labels
labels = to_categorical(labels, num_classes)

# split training from test data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_ratio, random_state=random_seed)

knn = neighbors.KNeighborsClassifier(n_neighbors=5, p=1, algorithm='brute')
knn.fit(x_train, y_train)

accuracy = knn.score(x_test, y_test)
print 'Accuracy: {}'.format(accuracy)

test_iris = np.array([5.3,3.1,2.2,0.1]) # 0: Iris-setosa
test_iris = test_iris.reshape(1, -1)

prediction = knn.predict(test_iris)
print 'Prediction: {}'.format(prediction)
