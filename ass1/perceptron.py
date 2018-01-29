import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, recall_score, precision_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["figure.figsize"] = 10,5
# %matplotlibinline

# load the iris dataset from sklearn
iris = datasets.load_iris()

# seperate features and targets
X = iris.data
y = iris.target

print(X, y)

# split data
test_size=0.3
random_state=0

# train_test_split convenience function
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=test_size,
                                                    random_state=random_state)

# standardize the data like we did before
sc = StandardScaler()
sc.fit(X_train)

# scale (transform) the training and the testing sets
# using the scaler that was fitted to training data
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# it is important to transform non-numeric target
# values into numbers prior to splitting the data
# to avoid unexpected results when modeling
print("Unique labels: {0}".format(np.unique(y)))

# we will select a subset of the features as before
X_train_std = X_train_std[:, [2,3]]
X_test_std = X_test_std[:, [2,3]]

# let's train a model using the sklearn
# implementation of perceptron
n_iter=40
eta0=0.1 #same as the other learning rate

# the instance
ppn = Perceptron(n_iter=n_iter, eta0=eta0, random_state=random_state)

ppn.fit(X_train_std, y_train)

# make predictions!
y_pred = ppn.predict(X_test_std)

# we can measure performance using the `accuracy_score`

print("precision: {0:.2f}%".format(precision_score(y_test, y_pred, average='micro') * 100))
print("recall: {0:.2f}%".format(recall_score(y_test, y_pred, average='micro') * 100))