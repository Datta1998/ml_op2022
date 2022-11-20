import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
gamma_list = [0.01, 0.008, 0.005, 0.001]
c_list = [0.1, 0.3, 0.5,0.7] 
h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]
from joblib import dump
from sklearn import svm, tree
import pdb

assert len(h_param_comb) == len(gamma_list)*len(c_list)
report = pd.DataFrame(h_param_comb)
train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1
digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images
data = digits.images.reshape((n_samples, -1))
dev_test_frac = 1-train_frac


best_acc = -1.0
best_model = None
best_h_params = None

svm_accuracy = []
# 2. For every combination-of-hyper-parameter values
Xtrain1, ytrain1, Xtest1, ytest1 = train_test_split(data, digits.target, test_size=0.1, random_state=30)

Xtrain2, ytrain2, Xtest2, ytest2 = train_test_split(data, digits.target, test_size=0.1, random_state=30)
Xtrain3, ytrain3, Xtest3, ytest3 = train_test_split(data, digits.target, test_size=0.1, random_state=70)
Xtrain4, ytrain4, Xtest4, ytest4 = train_test_split(data, digits.target, test_size=0.1, random_state=70)








def test_samecase():
    assert Xtrain1.all() == Xtrain2.all()
    assert Xtest1.all() == Xtest2.all()
    print("Successful")
def test_differentcase():
    assert (Xtest2 != Xtest3).any()
    assert (Xtrain2 != Xtrain3).any()
    print("Not Successful")

