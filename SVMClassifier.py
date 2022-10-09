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

model = svm.SVC()

SVCparams = { 'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
         'C': [1,10,20],
         'degree': [3,8],
         'coef0': [0.01,10,0.5],
         'gamma': ('auto','scale') }

grid_search = GridSearchCV(model, SVCparams, cv = 3,scoring="r2", verbose=2, n_jobs=-1)
random_search = RandomizedSearchCV(model, SVCparams, cv=3, n_iter=10, scoring="r2", verbose=2, n_jobs=-1, random_state=None)
bayes_search = BayesSearchCV(model, SVCparams, cv=3, n_iter=10, scoring="r2", verbose=2, n_jobs=-1, random_state=None)

for classification_dataset in classification_dataset_names[:5]:
    # Read in the datasets and split them into training/testing
    X, y = fetch_data(classification_dataset, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    grid_search.fit(X_train,y_train)
    print("Best Score for Grid Search is ", grid_search.best_score_)
    print("Best Parameters for Grid Search is ", grid_search.best_params_)

    random_search.fit(X_train,y_train)
    print("Best Params for dataset By Random Search are ", random_search.best_params_)
    print("Best Score for Random Search is ", random_search.best_score_)

    bayes_search.fit(X_train, y_train)
    print("Best Params for dataset By Bayes Search are ", bayes_search.best_params_)
    print("Best Score for Bayes Search is ", bayes_search.best_score_)