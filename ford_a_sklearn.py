import numpy as np
from datasets import readucr
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import sklearn.gaussian_process as gp
import sklearn.svm as svm
import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sktime.classification.shapelet_based import ROCKETClassifier
import time

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

all_classifier = [
    # ROCKETClassifier(),
    ensemble.RandomForestClassifier(),
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.BaggingClassifier(),
    neighbors.KNeighborsClassifier(),
    nn.MLPClassifier(),
    gp.GaussianProcessClassifier(),
    svm.LinearSVC(max_iter=15000),
    svm.NuSVC()
    
]

# x_train = np.expand_dims(x_train, axis=1)
# x_test = np.expand_dims(x_test, axis=1)

for clf in all_classifier:
    print(f"Starting training of {clf}")
    
    clf.fit(x_train, y_train)


    s = time.time()
    predictions = clf.predict(x_test)
    e = time.time()
    print(f"Inference took {e-s} s.")
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Accuracy of {clf}: {accuracy}")
    print(conf_matrix)