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

X_train_arr=[]
X_dev_arr=[]
y_train_arr=[]
y_dev_arr=[]

for i in range(0,5):
    
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )
    X_train_arr.append(X_train)
    X_dev_arr.append(X_dev)
    y_train_arr.append(y_train)
    y_dev_arr.append(y_dev)
    


from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(random_state=0)


svm_acc = []
dt_acc = []
cur_acc1=[]
dtcur_acc1=[]

# Learn the digits on the train subset
for i in range(0,5):

    clf.fit(X_train_arr[i], y_train_arr[i])
    predicted_dev = clf.predict(X_dev_arr[i])
    cur_acc=metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev_arr[i])
    print(cur_acc)
    cur_acc1.append(cur_acc)
    print(cur_acc1)




svm_acc.append(cur_acc1)


print("SVM accuracy: " ,svm_acc)
print("\nMax Acc: ",np.max(svm_acc))
print("\nMin acc: ",np.min(svm_acc))
print("\nMean acc: ",np.mean(svm_acc))
print("\nStd acc: ",np.std(svm_acc))


for i in range(5):

    clf1.fit(X_train_arr[i], y_train_arr[i])
    predicted_dev = clf1.predict(X_dev_arr[i])
    dtcur_acc=metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev_arr[i])
    print(dtcur_acc)
    dtcur_acc1.append(dtcur_acc)
    print(dtcur_acc1)


dt_acc.append(dtcur_acc1)

print("\n")
print("DTree accurcy: ", dt_acc)

print("\nMax Acc: ",np.max(dt_acc))
print("\nMin acc: ",np.min(dt_acc))
print("\nMean acc: ",np.mean(dt_acc))
print("\nStd acc: ",np.std(dt_acc))