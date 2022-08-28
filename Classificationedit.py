# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# them using :func:`matplotlib.pyplot.imread`.
#part: load dataset -- data from csv,tsv,json, pickle
digits = datasets.load_digits()

#part: sanity check of the data / visualisation

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#part: proprocessing -- to removesome noise, to normalise the data, format the data
#to be consumed by model
# flatten the images

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) #changing format of image
GAMMA=0.001
train_frac=0.8
dev_frac=0.1
test_frac=0.1
# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=GAMMA)

#Part setting up hyperparameters
clf.set_params(**hyper_params)

#Part: define train/dev/test splits of experiment protocol
# Split data into 50% train and 50% test subsets
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=1-train_frac, shuffle=False
)
X_test, X_dev, y_test, y_dev = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True
)
#if testing is same  as training set: the perfromance metrices may overestimate the goodness of the model
#you want to test on "unseen" samples 
#train to train the model
#dev to set hyperparameters of the model
#test to evaluate the performance of the model

# Learn the digits on the train subset
#creating instance of classifier class
#part: Define the model, setting up hyperparameters
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()