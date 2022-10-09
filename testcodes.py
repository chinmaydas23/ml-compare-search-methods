import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors, metrics
from pmlb import dataset_names, classification_dataset_names, regression_dataset_names
from pmlb import fetch_data
classfs = classification_dataset_names
print(classfs)
X,y = fetch_data('analcatdata_authorship', return_X_y=True, local_cache_dir='../datasets')
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
rf = RandomForestClassifier(random_state= None)
print(rf.get_params())
# Number of trees in random forest
ne = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
mf = ['sqrt']
# Maximum number of levels in tree
md = [int(x) for x in np.linspace(10, 110, num = 11)]
md.append(None)
# Minimum number of samples required to split a node
mss = [2, 5, 10]
# Minimum number of samples required at each leaf node
msl = [1, 2, 4]
# Method of selecting samples for training each tree
bs = [True, False]
# Create the random grid
random_grid = {'n_estimators': ne,
               'max_features': mf,
               'max_depth': md,
               'min_samples_split': mss,
               'min_samples_leaf': msl,
               'bootstrap': bs}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
# Random search of parameters, using 3 fold cross validation, search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model

rf_random.fit(X_train, y_train)
print(rf_random.best_score_)
print(rf_random.best_params_)