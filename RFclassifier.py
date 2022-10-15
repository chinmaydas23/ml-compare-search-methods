import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
from pmlb import dataset_names, classification_dataset_names, regression_dataset_names
from pmlb import fetch_data
import matplotlib.pyplot as plt


GridBestScore = []
RandBestScore = []
BayesBestScore = []

grid_search = GridSearchCV(RandomForestClassifier(random_state=0),
                          {
                              'n_estimators': np.arange(5, 100, 5),
                              'max_features': np.arange(0.1, 0.5, 0.05)
                          }, cv=3, scoring='accuracy', verbose=2, n_jobs=-1   # verbose values and accuracy scores
                          )
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=0),
                          {
                              'n_estimators': np.arange(5, 100, 5),
                              'max_features': np.arange(0.1, 0.5, 0.05)
                          }, n_iter=10, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
bayes_search = BayesSearchCV(RandomForestClassifier(random_state=0),
                          {
                              'n_estimators': np.arange(5, 100, 5),
                              'max_features': np.arange(0.1, 0.5, 0.05),
                          }, n_iter=10, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

labels = []
for classification_dataset in classification_dataset_names[5:15]:

    labels.append(classification_dataset)
    # Read in the datasets and split them into training/testing
    X, y = fetch_data(classification_dataset, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    grid_search.fit(X_train, y_train)
    print("Best Params for Grid Search on"+ classification_dataset + " are ", grid_search.best_params_)
    print("Best Score for Grid Search on " + classification_dataset + " is ", grid_search.best_score_)
    GridBestScore.append(grid_search.best_score_)

    random_search.fit(X_train, y_train)
    print("Best Params for dataset By Random Search are ", random_search.best_params_)
    print("Best Score for Random Search is ", random_search.best_score_)
    RandBestScore.append(random_search.best_score_)

    bayes_search.fit(X_train, y_train)
    print("Best Params for dataset By Bayes Search are ", bayes_search.best_params_)
    print("Best Score for Bayes Search is ", bayes_search.best_score_)
    BayesBestScore.append(bayes_search.best_score_)

print(GridBestScore)
print(RandBestScore)
print(BayesBestScore)

SMALL_SIZE = 6
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.plot(labels, GridBestScore, label="GridSearch")
plt.plot(labels, RandBestScore, label="RandSearch")
plt.plot(labels, BayesBestScore, label="BayesSearch")
plt.xlabel('Dataset Names -> ')
plt.ylabel('Scores -> ')
plt.title('COMPARISON OF SEARCH METHODS - RF CLASSIFIER')

# x = np.arange(len(labels))  # the label locations
# width = 0.40  # the width of the bars
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, GridBestScore, width, label='Grid Search')
# rects2 = ax.bar(x - width/6, RandBestScore, width, label='Random Search')
# rects3 = ax.bar(x + width/6, BayesBestScore, width, label='Bayesian Optimisation')
#
# ax.set_xlabel('Dataset Names -> ')
# ax.set_ylabel('Scores -> ')
# ax.set_title('COMPARISON OF SEARCH METHODS - RF CLASSIFIER')
# ax.set_xticks(x, labels)
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
#
# fig.tight_layout()
plt.legend()
plt.show()