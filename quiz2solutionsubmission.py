import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
import numpy as np
#resize(image, (100, 100)).shape(100, 100)

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.009, 0.007, 0.005, 0.001, 0.0008, 0.0005, 0.0004, 0.0002, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.5, 2, 5, 10]

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)


report = pd.DataFrame(h_param_comb)
print(report.head())


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)
#digits = datasets.load_digits()
data = digits.images
a  = data
#data = resize(data, (1797,10, 10))
#print('Resized image: ')
#print(data[0].shape)
data = digits.images.reshape((n_samples, -1))

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


best_acc = -1.0
best_model = None
best_h_params = None

#width, height = X_train.size

#print(width, height)


accuracy_train = []
accuracy_dev = []
accuracy_test = []
# 2. For every combination-of-hyper-parameter values
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
    predicted_train = clf.predict(X_train)
    predicted_dev = clf.predict(X_dev)
    predicted_test = clf.predict(X_test)

    # 2.b compute the accuracy on the validation set
    cur_acc1 = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
    cur_acc2 = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
    cur_acc3 = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
    accuracy_train.append(cur_acc1)
    accuracy_dev.append(cur_acc2)
    accuracy_test.append(cur_acc3)

    # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
    if cur_acc1 > best_acc:
        best_acc = cur_acc1
        best_model = clf
        best_h_params = cur_h_params
        print("Found new best acc with :"+str(cur_h_params))
        print("New best val accuracy:" + str(cur_acc1))
    if cur_acc2 > best_acc:
        best_acc = cur_acc2
        best_model = clf
        best_h_params = cur_h_params
        print("Found new best acc with :"+str(cur_h_params))
        print("New best val accuracy:" + str(cur_acc2))
    if cur_acc3 > best_acc:
        best_acc = cur_acc3
        best_model = clf
        best_h_params = cur_h_params
        print("Found new best acc with :"+str(cur_h_params))
        print("New best val accuracy:" + str(cur_acc3))



   
#PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(X_test)

#PART: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# 4. report the test set accurancy with that best model.
#PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

print("Best hyperparameters were:")
print(cur_h_params)
#print(accuracy)
accuracy1 = pd.DataFrame(accuracy_train,columns=['acc_train'])
accuracy2 = pd.DataFrame(accuracy_dev,columns=['acc_dev'])
accuracy3 = pd.DataFrame(accuracy_test,columns=['acc_test'])
final_report = pd.concat([report,accuracy1,accuracy2,accuracy3],axis=1)
#final_report
print("Size of the image: ",a[0].shape)
print(final_report)

print("Min, Max, Mean, Median of tge train set respectively are: ")
print(np.min(final_report['acc_train']),np.max(final_report['acc_train']), np.mean(final_report['acc_train']),np.median(final_report['acc_train']))

print("Min, Max, Mean, Median of the dev set respectively are: ")
print(np.min(final_report['acc_dev']),np.max(final_report['acc_dev']), np.mean(final_report['acc_dev']),np.median(final_report['acc_dev']))

print("Min, Max, Mean, Median of tge train set respectively are: ")
print(np.min(final_report['acc_test']),np.max(final_report['acc_test']), np.mean(final_report['acc_test']),np.median(final_report['acc_test']))