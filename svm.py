import pickle
import numpy as np
from sklearn import svm

with open("train.pkl","rb") as file:
    train_components,train_labels = pickle.load(file)

with open("test.pkl","rb") as file:
    test_components,test_labels = pickle.load(file)

test_components = np.transpose(test_components)
clf = svm.LinearSVC(multi_class="crammer_singer").fit(train_components,train_labels)

print ((1-clf.score(test_components,test_labels))*100)
