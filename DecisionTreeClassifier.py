import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
from pmlb import dataset_names, classification_dataset_names, regression_dataset_names
from pmlb import fetch_data

GridBestScore = []
RandBestScore = []
BayesBestScore = []

model = DecisionTreeClassifier(random_state=0)


DTCparams = {'criterion': ('gini', 'entropy'),
            'splitter': ('best', 'random'),
            'max_features': np.arange(0.1, 0.5, 0.05), }

grid_search = GridSearchCV(model, DTCparams, cv = 3, scoring='accuracy', verbose=2, n_jobs=-1, error_score='raise')
random_search = RandomizedSearchCV(model, DTCparams, n_iter=10, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
bayes_search = BayesSearchCV(model, DTCparams, n_iter=10, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

for classification_dataset in classification_dataset_names[:5]:
    # Read in the datasets and split them into training/testing
    X, y = fetch_data(classification_dataset, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    grid_search.fit(X_train,y_train)
    print("Best Score for Grid Search on " + classification_dataset + " is ", grid_search.best_score_)
    print("Best Parameters for Grid Search on "+ classification_dataset + " is ", grid_search.best_params_)
    GridBestScore.append(grid_search.score(X_test, y_test))

    random_search.fit(X_train, y_train)
    print("Best Params for dataset By Random Search are ", random_search.best_params_)
    print("Best Score for Random Search is ", random_search.best_score_)
    RandBestScore.append(random_search.score(X_test, y_test))

    bayes_search.fit(X_train, y_train)
    print("Best Params for dataset By Bayes Search are ", bayes_search.best_params_)
    print("Best Score for Bayes Search is ", bayes_search.best_score_)
    BayesBestScore.append(bayes_search.score(X_test,y_test))

print(GridBestScore)
print(RandBestScore)
print(BayesBestScore)