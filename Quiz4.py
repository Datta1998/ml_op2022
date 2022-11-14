import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
gamma_list = [0.01, 0.008, 0.005, 0.001]
c_list = [0.1, 0.3, 0.5, 0.7] 
h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]
from joblib import dump
from sklearn import svm, tree
import pdb

assert len(h_param_comb) == len(gamma_list)*len(c_list)
report = pd.DataFrame(h_param_comb)
train_frac = 0.8
test_frac = 0.1
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
for i in range(0,5):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )
    for cur_h_params in h_param_comb:

        #PART: Define the model
        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)


        #PART: Train model
        # 2.a train the model 
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # print(cur_h_params)
        #PART: get dev set predictions
        predicted_dev = clf.predict(X_test)

        # 2.b compute the accuracy on the validation set
        cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_test)
        

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_model = clf
            best_h_params = cur_h_params
    svm_accuracy.append(cur_acc)
            


clf = svm.SVC()
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)
hyper_params = best_h_params
clf.set_params(**best_h_params)
clf.fit(X_train,y_train)
ypred = clf.predict(X_test)
#rint('Predicted labels for SVM after hyperparameter tuning')
report = pd.DataFrame()
report['Actual digits'] = y_test
report['Predicted digits'] = ypred
#print(report.head(10))
best_param_config = "_".join(
        [h + "=" + str(best_h_params[h]) for h in best_h_params]
    )
dump(clf,"svm" + "_" + best_param_config + ".joblib")