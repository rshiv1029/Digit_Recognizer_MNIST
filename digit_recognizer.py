# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import sys
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv, concat, Series
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# print(os.listdir("../input/digit-recognizer/"))

train_name = "./train.csv"
train = read_csv(train_name)

test_name = "./test.csv"
test = read_csv(test_name)

print("Training set dimensions: " + str(train.shape))
print("Testing set dimensions: " + str(test.shape))

X = train.drop("label", axis=1)
y = train["label"]
X.shape

sc = StandardScaler()
X = sc.fit_transform(X)
test = sc.transform(test)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(y[i])
    plt.imshow(X[i, :].reshape(28, 28))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


print("X_train shape is : ", X_train.shape)
print("Y_train shape is : ", y_train.shape)
print("X_test shape is ", X_test.shape)
print("Y_test shape is ", y_test.shape)

logreg = LogisticRegression(random_state=1, max_iter=150, solver="sag", tol=0.1)
logreg.fit(X_train, y_train)

predictions = logreg.predict(X_test)
predictions

result = logreg.predict(test)
result = Series(result, name="Label")
result.shape
submission = concat([Series(range(1, 28001), name="ImageId"), result], axis=1)
submission.to_csv("mnist_submit.csv", index=False)

print("train accuracy: {} ".format(logreg.score(X_train, y_train)))
print("test accuracy: {} ".format(logreg.score(X_test, y_test)))
