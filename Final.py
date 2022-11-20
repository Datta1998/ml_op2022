import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

gamma_list = [0.01, 0.008, 0.005]
c_list = [0.1, 0.3, 0.5] 
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
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True, random_state=30
    )
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True, random_state=30
    )
for cur_h_params in h_param_comb:
    clf = svm.SVC()
    hyper_params = cur_h_params
    clf.set_params(**hyper_params)
    clf.fit(X_train, y_train)
    predicted_dev = clf.predict(X_test)
    cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_test)
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_model = clf
        best_h_params = cur_h_params
svm_accuracy.append(cur_acc)
hyper_params = best_h_params
clf.set_params(**best_h_params)
clf.fit(X_train,y_train)
ypred = clf.predict(X_test)
#print('Predicted labels for SVM after hyperparameter tuning')
report = pd.DataFrame()
report['Actual digits'] = y_test
report['Predicted digits'] = ypred
#print(report)
a=accuracy_score(y_test, ypred)
print("test accuracy:",a)
b=f1_score(y_test.reshape(-1,1), ypred.reshape(-1,1), average='macro')
print("test macro-f1:",b)
random_state=30
state=random_state

best_param_config = "_".join(
        [h + "=" + str(best_h_params[h]) for h in best_h_params]
    )
#dump(clf,"svm" + "_" + best_param_config + "random_state" + str(state) + ".joblib")
dump(clf,"Models/"+"Svm" + "_" + str(best_param_config) +"Random_state: "+str(state)+ ".joblib")
print("model saved at ./models/svm_gamma=0.0005_C=0.5.joblib")









