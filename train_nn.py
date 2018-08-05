import numpy as np
import pandas as pd

from keras.utils import to_categorical
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

data_file = 'data/iris.data'
label_name = 'class'
num_classes = 3
test_ratio = 0.1
random_seed = 2
feature_count = 4
epoch = 50
batch_size = 10

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

model = Sequential()
model.add(Dense(3, input_dim=feature_count, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=test_ratio)

scores = model.evaluate(x_test, y_test)
print 'Test Loss: {}, Test Accuracy {}'.format(scores[0], scores[1])

test_iris = np.array([5.3,3.1,2.2,0.1]) # 0: Iris-setosa
test_iris = test_iris.reshape(1, -1)
predictions = model.predict(test_iris)
import operator
(prediction, confidence) = max(enumerate(predictions[0]), key=operator.itemgetter(1))
print 'Prediction: {}, Confidence: {}'.format(prediction, confidence)
