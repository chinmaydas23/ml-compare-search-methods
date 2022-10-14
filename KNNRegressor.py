import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
from pmlb import dataset_names, classification_dataset_names, regression_dataset_names
from pmlb import fetch_data

GridBestScore = []
RandBestScore = []
BayesBestScore = []

model = svm.SVR()

KNRparams = {   'weights' : ('uniform', 'distance'),
                'n_neighbours' : np.arange(1,5,1),
                'algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
            }

grid_search = GridSearchCV(model, KNRparams, cv = 3,scoring="r2", verbose=2, n_jobs=-1)
random_search = RandomizedSearchCV(model, KNRparams, n_iter = 5, cv = 3, verbose=2, random_state=None, n_jobs = -1)
bayes_search = BayesSearchCV(model, KNRparams, cv=3, n_iter=5, scoring="r2", verbose=2, n_jobs=-1, random_state=None)

for regression_dataset in regression_dataset_names[:5]:
    # Read in the datasets and split them into training/testing
    X, y = fetch_data(regression_dataset, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    grid_search.fit(X_train,y_train)
    print("Best Score for Grid Search on " + regression_dataset + " is ", grid_search.best_score_)
    print("Best Parameters for Grid Search on "+ regression_dataset + " is ", grid_search.best_params_)

    random_search.fit(X_train,y_train)
    print("Best Params for dataset By Random Search are ", random_search.best_params_)
    print("Best Score for Random Search is ", random_search.best_score_)

    bayes_search.fit(X_train, y_train)
    print("Best Params for dataset By Bayes Search are ", bayes_search.best_params_)
    print("Best Score for Bayes Search is ", bayes_search.best_score_)