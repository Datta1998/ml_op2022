import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.002, 0.0007, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 0.9, 1, 2, 5, 7, 10]
h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

h1 = pd.DataFrame(h_param_comb)
#print(report.head())

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
data11=digits.images
data1 = digits.images
print(data1[0].shape)
data1 = resize(data1, (1797,20, 20))
a=data1

data1 = digits.images.reshape((n_samples, -1))

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data1, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


best_acc = -1.0
best_model = None
best_h_params = None

#width, height = X_train.size

#print(width, height)
accuracy1_train = []
accuracy1_dev = []
accuracy1_test = []
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
    predicted_train1 = clf.predict(X_train)
    predicted_dev1 = clf.predict(X_dev)
    predicted_test1 = clf.predict(X_test)

    # 2.b compute the accuracy on the validation set
    cur_acc1 = metrics.accuracy_score(y_pred=predicted_train1, y_true=y_train)
    cur_acc2 = metrics.accuracy_score(y_pred=predicted_dev1, y_true=y_dev)
    cur_acc3 = metrics.accuracy_score(y_pred=predicted_test1, y_true=y_test)
    accuracy1_train.append(cur_acc1)
    accuracy1_dev.append(cur_acc2)
    accuracy1_test.append(cur_acc3)

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
accuracy1 = pd.DataFrame(accuracy1_train,columns=['acc_train'])
accuracy2 = pd.DataFrame(accuracy1_dev,columns=['acc_dev'])
accuracy3 = pd.DataFrame(accuracy1_test,columns=['acc_test'])
final_report = pd.concat([h1,accuracy1,accuracy2,accuracy3],axis=1)
#final_report
print('Original Image Size: ')
print(data11[0].shape)
print('Resized image: ')
print(a[0].shape)
print(final_report)


#2nd_image
h2 = pd.DataFrame(h_param_comb)
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
data22=digits.images
data2 = digits.images
print('Image Size')
print(data2[0].shape)
data2 = resize(data2, (1797,40, 40))
b=data2

data2 = digits.images.reshape((n_samples, -1))

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data2, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


best_acc = -1.0
best_model = None
best_h_params = None

#width, height = X_train.size

#print(width, height)
accuracy2_train = []
accuracy2_dev = []
accuracy2_test = []
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
    predicted_train2 = clf.predict(X_train)
    predicted_dev2 = clf.predict(X_dev)
    predicted_test2 = clf.predict(X_test)

    # 2.b compute the accuracy on the validation set
    cur_acc1 = metrics.accuracy_score(y_pred=predicted_train2, y_true=y_train)
    cur_acc2 = metrics.accuracy_score(y_pred=predicted_dev2, y_true=y_dev)
    cur_acc3 = metrics.accuracy_score(y_pred=predicted_test2, y_true=y_test)
    accuracy2_train.append(cur_acc1)
    accuracy2_dev.append(cur_acc2)
    accuracy2_test.append(cur_acc3)

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
accuracy1 = pd.DataFrame(accuracy2_train,columns=['acc_train'])
accuracy2 = pd.DataFrame(accuracy2_dev,columns=['acc_dev'])
accuracy3 = pd.DataFrame(accuracy2_test,columns=['acc_test'])
final_report2 = pd.concat([h2,accuracy1,accuracy2,accuracy3],axis=1)
#final_report
print('Original Image Size: ')
print(data22[0].shape)
print('Resized image: ')
print(b[0].shape)
print(final_report2)

#thirdResolution

d3 = pd.DataFrame(h_param_comb)

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
data33=digits.images
data3 = digits.images
print('Image Size')
print(data3[0].shape)
data3 = resize(data3, (1797,60, 60))
c=data3

data3 = digits.images.reshape((n_samples, -1))

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data3, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


best_acc = -1.0
best_model = None
best_h_params = None

#width, height = X_train.size

#print(width, height)
accuracy3_train = []
accuracy3_dev = []
accuracy3_test = []
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
    predicted_train3 = clf.predict(X_train)
    predicted_dev3 = clf.predict(X_dev)
    predicted_test3 = clf.predict(X_test)

    # 2.b compute the accuracy on the validation set
    cur_acc1 = metrics.accuracy_score(y_pred=predicted_train3, y_true=y_train)
    cur_acc2 = metrics.accuracy_score(y_pred=predicted_dev3, y_true=y_dev)
    cur_acc3 = metrics.accuracy_score(y_pred=predicted_test3, y_true=y_test)
    accuracy3_train.append(cur_acc1)
    accuracy3_dev.append(cur_acc2)
    accuracy3_test.append(cur_acc3)

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
accuracy1 = pd.DataFrame(accuracy3_train,columns=['acc_train'])
accuracy2 = pd.DataFrame(accuracy3_dev,columns=['acc_dev'])
accuracy3 = pd.DataFrame(accuracy3_test,columns=['acc_test'])
final_report3 = pd.concat([d3,accuracy1,accuracy2,accuracy3],axis=1)
#final_report
print('Original Image Size: ')
print(data33[0].shape)
print('Resized image: ')
print(c[0].shape)
print(final_report3)

