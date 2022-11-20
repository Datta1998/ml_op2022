import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import argparse 
from sklearn.metrics import f1_score,accuracy_score
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump
import pdb
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images
data = digits.images.reshape((n_samples, -1))


parser = argparse.ArgumentParser(description = 'Testing')
parser.add_argument('--clf_name', type = str)

parser.add_argument('--random_state', type = int)
args = parser.parse_args()

clf = args.clf_name
random_state = args.random_state

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.30, random_state=random_state)
modelset = {'dtree': DecisionTreeClassifier(max_depth=10),'svm': svm.SVC(gamma=0.05,C=0.1)}

def trainingmodel_random_state(clf,random_state):
    clf = modelset[clf]
    clf.fit(X_train,y_train)
    ypred = clf.predict(X_test)
    b = f1_score(y_test.reshape(-1,1), ypred.reshape(-1,1), average='macro')
    a = accuracy_score(y_test.reshape(-1,1), ypred.reshape(-1,1))
    print("Accuracy: ",a)
    print("Macro-f1: ",b)

trainingmodel_random_state(clf,random_state)