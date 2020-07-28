import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"          # label

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# splitting data into 4 different arrays
# test variables are used to test the accuracy of model
# can't train model off testing data because it has already seen that information - nothing new learned
# test sample is split up by 10 percent

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# # looking for best training of model - trying to get the highest accuracy
# best = 0
# for _ in range(30):
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     # creating linear regression model
#     linear = linear_model.LinearRegression()
#     # finds best fit line
#     linear.fit(x_train, y_train)
#     # returns accuracy of model
#     acc = linear.score(x_test, y_test)
#     print(acc)
#
#     if acc > best:
#         best = acc
#         # saving model so you don't have to always train every run
#         with open("student_model.pickle", "wb") as f:      # wb mode writes the file in case the file doesn't exist
#             pickle.dump(linear, f)                         # dump linear model into the file f

pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)


print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# predict on a student. did not use test data to train our model
predictions = linear.predict(x_test)

for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# plotting data - not ML predictions
p = "failures"                          # x axis
style.use("ggplot")                     # makes grid look better
pyplot.scatter(data[p], data["G3"])     # setting x and why
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()