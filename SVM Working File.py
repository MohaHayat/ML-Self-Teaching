import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# load in data
cancer = datasets.load_breast_cancer()          # pick a dataset that is classifiable for SVM

x = cancer.data                                 # features of data set 
y = cancer.target                               # 0 or 1 - malignant or benign - classification
# is test_size the validation set?
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# classes SVM will classify to
classes = ['malignant', 'benign']       # 0 and 1 will correspond to these

# create classification
# SVC = support vector classification
clf = svm.SVC(kernel="linear", C=1)                     # C enables a soft margin - 0 is hard
clf.fit(x_train, y_train)

# predict data before you can score
y_predict = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_predict)

print(acc)

