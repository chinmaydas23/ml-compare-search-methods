import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from pmlb import dataset_names, classification_dataset_names, regression_dataset_names
from pmlb import fetch_data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

GridTestScore = []
RandTestScore = []
BayesTestScore = []

GridTimeTaken = []
RandTimeTaken = []
BayesTimeTaken = []

# Creating the parameter grids of each model. These parameters will get optimized
RFparams = {'n_estimators': np.arange(5, 100, 5),
            'max_features': np.arange(0.1, 0.5, 0.05)}
DTCparams = {'criterion': ('gini', 'entropy'),
            'splitter': ('best', 'random'),
            'max_features': np.arange(0.1, 0.5, 0.05) }
DTRparams = {'criterion': ('squared_error', 'friedman_mse', 'absolute_error', 'poisson'),
            'splitter': ('best', 'random'),
            'max_features': np.arange(0.1, 0.5, 0.05), }
KNparams = {'weights': ('uniform', 'distance'),
                # 'n_neighbours' : np.arange(1,5,1),
                'algorithm' : ('auto', 'ball_tree', 'kd_tree', 'brute')
            }
SVparams = { 'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
         'C': [1,10,20],
         'degree': [3,8],
         'coef0': [0.01,10,0.5],
         'gamma': ('auto','scale') }

modelnames = ["RF CLASSIFIER", "SVM CLASSIFIER", "DT CLASSIFIER", "KNN CLASSIFIER", "RF REGRESSOR", "SVM REGRESSOR",
              "DT REGRESSOR", "KNN REGRESSOR"]

#Selecting the model out of the four:
for i in range(8):
    if i == 0:
        model = RandomForestClassifier(random_state=0)
        params = RFparams
    elif i == 1:
        model = svm.SVC()
        params = SVparams
    elif i == 2:
        model = DecisionTreeClassifier(random_state=0)
        params = DTCparams
    elif i == 3:
        model = KNeighborsClassifier()
        params = KNparams
    elif i == 4:
        model = RandomForestRegressor(random_state=0)
        params = RFparams
    elif i == 5:
        model = svm.SVR()
        params = SVparams
    elif i == 6:
        model = DecisionTreeRegressor(random_state=0)
        params = DTRparams
    else:
        model = KNeighborsRegressor()
        params = KNparams

    if 0 <= i <= 3:
        # CLASSIFICATION Models:

        grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy', verbose=2, n_jobs=-1, error_score='raise')
        random_search = RandomizedSearchCV(model, params, n_iter=10, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        bayes_search = BayesSearchCV(model, params, n_iter=10, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

        for classification_dataset in classification_dataset_names:
            # Read in the datasets and split them into training/testing
            X, y = fetch_data(classification_dataset, return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
    elif i >= 4:
        # REGRESSION models:

        grid_search = GridSearchCV(model, params, cv=3, scoring="r2", verbose=2, n_jobs=-1,error_score='raise')
        random_search = RandomizedSearchCV(model, params, n_iter=5, cv=3, verbose=2, random_state=None, n_jobs=-1)
        bayes_search = BayesSearchCV(model, params, cv=3, n_iter=5, scoring="r2", verbose=2, n_jobs=-1,random_state=None)

        for regression_dataset in regression_dataset_names:
            X, y = fetch_data(regression_dataset, return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y)

    start_time = datetime.now()
    grid_search.fit(X_train, y_train)
    print("Best Params for Grid Search using "+modelnames[i]+" model are:", grid_search.best_params_)
    print("Best Score for Grid Search using "+modelnames[i]+" model is:", grid_search.best_score_)
    GridTestScore.append(grid_search.score(X_test,y_test))
    end_time = datetime.now()
    duration = str(end_time - start_time)
    print('Duration for Grid Search: {}'.format(end_time - start_time))
    GridTimeTaken.append(duration)

    start_time = datetime.now()
    random_search.fit(X_train, y_train)
    print("Best Params for dataset By Random Search using "+modelnames[i]+" model are:", random_search.best_params_)
    print("Best Score for Random Search using "+modelnames[i]+" model is:", random_search.best_score_)
    RandTestScore.append(random_search.score(X_test,y_test))
    end_time = datetime.now()
    duration = str(end_time - start_time)
    print('Duration for Random Search: {}'.format(end_time - start_time))
    RandTimeTaken.append(duration)

    start_time = datetime.now()
    bayes_search.fit(X_train, y_train)
    print("Best Params for dataset By Bayes Search using "+modelnames[i]+" model are:", bayes_search.best_params_)
    print("Best Score for Bayes Search using "+modelnames[i]+" model is:", bayes_search.best_score_)
    BayesTestScore.append(bayes_search.score(X_test,y_test))
    end_time = datetime.now()
    duration = str(end_time - start_time)
    print('Duration for Bayes Search: {}'.format(end_time - start_time))
    BayesTimeTaken.append(duration)

    # Getting the average value of the scores list for each search method
    gavg = round(sum(GridTestScore)/len(GridTestScore), 5)
    ravg = round(sum(RandTestScore)/len(RandTestScore), 5)
    bavg = round(sum(BayesTestScore)/len(BayesTestScore), 5)
    print(gavg)
    print(ravg)
    print(bavg)

    # Getting the average value of the time durations list for each search method
    gtavg = (timedelta(seconds=sum(map(lambda f: float(f[0])*3600 + float(f[1])*60 + float(f[2]), map(lambda f: f.split(':'), GridTimeTaken)))/len(GridTimeTaken)))
    rtavg = (timedelta(seconds=sum(map(lambda f: float(f[0])*3600 + float(f[1])*60 + float(f[2]), map(lambda f: f.split(':'), RandTimeTaken)))/len(RandTimeTaken)))
    btavg = (timedelta(seconds=sum(map(lambda f: float(f[0])*3600 + float(f[1])*60 + float(f[2]), map(lambda f: f.split(':'), BayesTimeTaken)))/len(RandTimeTaken)))
    print(gtavg)
    print(rtavg)
    print(btavg)

    # Getting the plotting parameters correct and ready
    SMALL_SIZE = 6
    MEDIUM_SIZE = 10
    BIG_SIZE = 14
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIG_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Defining the axes and values thereof
    xlabels = ['GRID SEARCH', 'RANDOM SEARCH', 'BAYESIAN OPTIMISATION']

    # Getting the value of each bar to show on top of it
    def add_value_labels(ax, spacing=5):
        """Add labels to the end of each bar in a bar chart."""

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            va = 'bottom'
            # Use Y value as label and format number with four decimal places
            label = "{:.5f}".format(y_value)
            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, spacing),              # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label


    width = 0.40  # the width of the bars
    avgscores = [gavg, ravg, bavg]
    avg_series1 = pd.Series(avgscores)
    plt.figure(figsize=(12,8))
    ax = avg_series1.plot(kind='bar', color= ['red', 'green', 'blue'])
    ax.set_xlabel('Search Method -> ')
    ax.set_ylabel('Average Score -> ')
    ax.set_title("COMPARISON OF SEARCH METHODS - Accuracy Scores: "+modelnames[i]+" ")
    ax.set_xticklabels(xlabels, rotation=0)
    add_value_labels(ax)
    plt.savefig(" AvgScoresPlot " + modelnames[i] + ".pdf", dpi=100)
    plt.show()

    avgtimes = [gtavg.total_seconds(), rtavg.total_seconds(), btavg.total_seconds()]
    avg_series2 = pd.Series(avgtimes)
    plt.figure(figsize=(12,8))
    ax2 = avg_series2.plot(kind='bar', color= ['red', 'green', 'blue'])
    ax2.set_xlabel('Search Method -> ')
    ax2.set_ylabel('Average Time (in seconds)-> ')
    ax2.set_title("COMPARISON OF SEARCH METHODS - Time of Execution: "+modelnames[i] +" ")
    ax2.set_xticklabels(xlabels, rotation=0)
    add_value_labels(ax2)
    plt.savefig(" AvgTimesPlot " + modelnames[i] + ".pdf", dpi=100)
    plt.show()