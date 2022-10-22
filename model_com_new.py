import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1



n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train1, X_dev_test1, y_train1, y_dev_test1 = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test1, X_dev1, y_test1, y_dev1 = train_test_split(
    X_dev_test1, y_dev_test1, test_size=(dev_frac)/dev_test_frac, shuffle=True
)

X_train2, X_dev_test2, y_train2, y_dev_test2 = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test2, X_dev2, y_test2, y_dev2 = train_test_split(
    X_dev_test2, y_dev_test2, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


X_train3, X_dev_test3, y_train3, y_dev_test3 = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test3, X_dev3, y_test3, y_dev3 = train_test_split(
    X_dev_test3, y_dev_test3, test_size=(dev_frac)/dev_test_frac, shuffle=True
)

X_train4, X_dev_test4, y_train4, y_dev_test4 = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test4, X_dev4, y_test4, y_dev4 = train_test_split(
    X_dev_test1, y_dev_test1, test_size=(dev_frac)/dev_test_frac, shuffle=True
)



X_train5, X_dev_test5, y_train5, y_dev_test5 = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test5, X_dev5, y_test5, y_dev5 = train_test_split(
    X_dev_test5, y_dev_test5, test_size=(dev_frac)/dev_test_frac, shuffle=True
)



from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(random_state=0)






svm_acc = []
dt_acc = []

# Learn the digits on the train subset


clf.fit(X_train1, y_train1)
predicted_dev1 = clf.predict(X_dev1)
clf.fit(X_train2, y_train2)
predicted_dev2 = clf.predict(X_dev2)
clf.fit(X_train3, y_train3)
predicted_dev3 = clf.predict(X_dev3)
clf.fit(X_train4, y_train4)
predicted_dev4 = clf.predict(X_dev4)
clf.fit(X_train5, y_train5)
predicted_dev5 = clf.predict(X_dev5)

cur_acc1 = metrics.accuracy_score(y_pred=predicted_dev1, y_true=y_dev1)
cur_acc2 = metrics.accuracy_score(y_pred=predicted_dev2, y_true=y_dev2)
cur_acc3 = metrics.accuracy_score(y_pred=predicted_dev3, y_true=y_dev3)
cur_acc4 = metrics.accuracy_score(y_pred=predicted_dev4, y_true=y_dev4)
cur_acc5 = metrics.accuracy_score(y_pred=predicted_dev5, y_true=y_dev5)


cur_acc=[cur_acc1,cur_acc2,cur_acc3,cur_acc4,cur_acc5]


svm_acc.append(cur_acc)


print("SVM accuracy: " ,svm_acc)
print("\nMax Acc: ",np.max(svm_acc))
print("\nMin acc: ",np.min(svm_acc))
print("\nMean acc: ",np.mean(svm_acc))
print("\nStd acc: ",np.std(svm_acc))




clf1.fit(X_train1, y_train1)


# Predict the value of the digit on the test subset
predicted_dev1 = clf1.predict(X_dev1)
clf.fit(X_train2, y_train2)
predicted_dev2 = clf1.predict(X_dev2)
clf.fit(X_train3, y_train3)
predicted_dev3 = clf1.predict(X_dev3)
clf.fit(X_train4, y_train4)
predicted_dev4 = clf1.predict(X_dev4)
clf.fit(X_train5, y_train5)
predicted_dev5 = clf1.predict(X_dev5)

cur_acc1 = metrics.accuracy_score(y_pred=predicted_dev1, y_true=y_dev1)
cur_acc2 = metrics.accuracy_score(y_pred=predicted_dev2, y_true=y_dev2)
cur_acc3 = metrics.accuracy_score(y_pred=predicted_dev3, y_true=y_dev3)
cur_acc4 = metrics.accuracy_score(y_pred=predicted_dev4, y_true=y_dev4)
cur_acc5 = metrics.accuracy_score(y_pred=predicted_dev5, y_true=y_dev5)

dt_acc=[cur_acc1,cur_acc2,cur_acc3,cur_acc4,cur_acc5]


svm_acc.append(dt_acc)

print("\n")
print("DTree accurcy: ", dt_acc)

print("\nMax Acc: ",np.max(dt_acc))
print("\nMin acc: ",np.min(dt_acc))
print("\nMean acc: ",np.mean(dt_acc))
print("\nStd acc: ",np.std(dt_acc))