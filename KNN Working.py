import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# reading in data
data = pd.read_csv("car.data")

# setting up data from  non numerical to numerical  - making list of ints and then a complete table

# need to make all our non integer values into integers
le = preprocessing.LabelEncoder()                           # encodes labels into appropriate integer values
buying = le.fit_transform(list(data["buying"]))             # gets entire buying column of data set
maint = le.fit_transform(list(data["maint"]))               # returns a numpy array
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
car_class = le.fit_transform(list(data["class"]))

x = list(zip(buying, maint, door, persons, lug, safety))    # zips all labels into one list - creates a bunch of tuples
y = list(car_class)

# smaller test size is better because it is leaves more data to train on thus higher accuracy
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# setup KNN
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

# predict using linear model
predict = model.predict(x_test)

# testing trained model vs test results
# classifying based on what the dataset uses
names = ["unacc", "acc", "good", "vgood"]           # right now it is going 0 to 3, but now we get the actual value

for x in range(len(predict)):                       # use test for index
    print("Predicted: ", names[predict[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

    # finds the k-neighbors of a point and returns the indices distances of each point
    n = model.kneighbors([x_test[x]], 5, True)          # using [[]] comes in as 2D when we only want to send in 1D
    print("N: ", n[0])






