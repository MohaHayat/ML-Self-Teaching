import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# load in data
digits = load_digits()

# .data is all of our features, scale scales each ft down so its between -1 and 1
# scaling makes things faster
data = scale(digits.data)
y = digits.target

# set # of clusters/centroids to make
# k = len(np.unique(y)) -  dynamic way to do it.
k = 10                      # because we have 10 digits, ie 10 classes

# get numbers of instances and features
samples, features = data.shape


# scoring the model - taken from sklearn
# since this is an unsupervised model, you don't need test data
# so you would compare the predicted classifiers (labels) to the targets we have
# it automatically generates its own y value when it runs/trains.
# look at notes for what each score means
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# k clusters/centroids, init clusters can be random or structured on a grid, n_init number of times the algo will
# run with different centroid starting placements - returns best one
classifier = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(classifier, '1', data)
