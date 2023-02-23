from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import random

""" ___ PREDICTION OF SPECIES OF IRIS FLOWER USING LOGISTIC REGRESSION ___ """

"""Load the Iris plants dataset"""
iris = load_iris()

"""Create a Pandas DataFrame for the datase"""
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].apply(lambda x: iris.target_names[x])
# iris_df contains the features and target values for each sample in the dataset

"""Print the first 10 rows of the DataFrame"""
# print(iris_df[:10])  # print(iris_df.head(10))

"""Split the dataset into 70% training and 30% testing sets"""
seed = random.randint(1, 1000)
# seed = 42
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=seed)

"""Train a logistic regression model on the training set"""
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

"""Evaluate the performance of the model on the testing set"""
y_predict = model.predict(x_test)
# Evaluate the performance of the model on the testing set
accuracy = accuracy_score(y_test, y_predict)
# print the accuracy of the model in predicting the species of iris flowers
# print("Accuracy: {:.2f}%".format(accuracy * 100))

"""Use k-fold cross-validation to evaluate the performance of the model"""
scores = cross_val_score(model, iris.data, iris.target, cv=10)
# print the mean accuracy of the model across all folds
# print("Mean accuracy: {:.2f}%".format(scores.mean() * 100))

"""Calculate evaluation metrics"""
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1:.2f}")

"""Create a confusion matrix for the model"""
confusion = confusion_matrix(y_test, y_predict)
# shows the number of true positives, true negatives, false positives, and false negatives for each class
print("Confusion matrix:")
print(confusion)
