import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Read in the data using pandas from the files
mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")

# print(mnist_train)
# Upon printing the dataset, it is seen that there is a label column for each image that shows which digit it is,
# and there are 28x28 pixel intensities for each.
# Now, we separate the label column from the rest of the train dataset so we only have the pixel intensities
# They are also converted to numpy arrays for simplicity

train_labels = mnist_train["label"].copy().to_numpy()
X_train = mnist_train.drop(columns=["label"]).to_numpy()

# We do the same for the test dataset

test_labels = mnist_test["label"].copy().to_numpy()
X_test = mnist_test.drop(columns=["label"]).to_numpy()

# Beginning the implementation of the KNN algorithm
knn = KNeighborsClassifier()

# Fitting the data to the model
knn.fit(X_train, train_labels)

# Evaluating the model through cross-validation
knn_preds = cross_val_predict(knn, X_train, train_labels, cv=3)

# Use Grid Search to determine the optimal k-value
knn_grid_search = GridSearchCV(knn, param_grid=[{"weights": ["uniform", "distance"],"n_neighbors": [2, 3, 5, 7]}], cv=3, scoring="accuracy", n_jobs=2)
knn_grid_search.fit(X_train, train_labels)
print(knn_grid_search.best_estimator_)

# Update the model according to the best value of k determined
updated_model = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(X_train, train_labels)

# Use the updated model to predict the values of the test dataset
test_preds = updated_model.predict(X_test)

# Print the model report of the classification on the test set
print(classification_report(test_labels, test_preds, digits=6))





