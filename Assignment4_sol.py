import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
gamma_list = [0.01, 0.008]
c_list = [0.1, 0.3] 
h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

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
print('Predicted labels for SVM after hyperparameter tuning')
report = pd.DataFrame()
report['Actual digits'] = y_test
report['Predicted digits'] = ypred
print(report.head(10))




best_acc = -1.0
best_model = None
best_h_params = None


from sklearn.tree import DecisionTreeClassifier
max_depth = [5, 10,15, 20]
min_samples_split = [2,4,5,7] 

h_param_combdt = [{'max_depth':g, 'min_samples_split':c} for g in max_depth for c in min_samples_split]

assert len(h_param_combdt) == len(max_depth)*len(min_samples_split)

dt_accuracy = []


for i in range(0,5):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )
    for cur_h_paramsdt in h_param_combdt:

        #PART: Define the model
        # Create a classifier: a support vector classifier
        clf = DecisionTreeClassifier()

        #PART: setting up hyperparameter
        hyper_paramsdt = cur_h_paramsdt
        clf.set_params(**hyper_paramsdt)


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
            best_h_paramsdt = cur_h_paramsdt
    dt_accuracy.append(cur_acc)
   
clf1 = DecisionTreeClassifier()
X_train1, X_dev_test1, y_train1, y_dev_test1 = train_test_split(
data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test1, X_dev1, y_test1, y_dev1 = train_test_split(
X_dev_test1, y_dev_test1, test_size=(dev_frac)/dev_test_frac, shuffle=True
)
hyper_paramsdt = best_h_paramsdt
clf1.set_params(**best_h_paramsdt)
clf1.fit(X_train1,y_train1)
ypred1 = clf1.predict(X_test1)
print('Predicted labels for Decision Tree after hyperparameter tuning')
report1 = pd.DataFrame()
report1['Actual digits'] = y_test1
report1['Predicted digits'] = ypred1
print(report1.head(10))





print("\n")
print("Accuracy of SVM are as follows:", svm_accuracy)
print("Accuracy of Decision Tree are as follows:", dt_accuracy)
avg1=np.average(svm_accuracy)
avg2=np.average(dt_accuracy)
avg=[avg1,avg2]
model=['SVM','Decision Tree']
print("\n")
print("Mean and standard Deviation value of each models:")
print("\n")
Mean=[np.mean(svm_accuracy),np.mean(dt_accuracy)]
std=[np.std(svm_accuracy),np.std(dt_accuracy)]
a=pd.DataFrame()
a['Model']=model
a['Mean']=Mean
a['Standard Deviation']=std
a['Accuracy Average']=avg
print(a)