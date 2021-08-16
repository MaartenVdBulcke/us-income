import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, SCORERS, matthews_corrcoef, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    train_test_split, KFold, RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE

from utils.evaluate import *
from utils.dataset_manipulation import *
from utils.plotting import *

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

income_train = pd.read_csv(r'datasets\US Income\cleaned\data_train.csv')
income_test = pd.read_csv(r'datasets\US Income\cleaned\data_test.csv')

if __name__ == '__main__':
    X_train, y_train = split_features_and_target(income_train, 'income')
    X_test, y_test = split_features_and_target(income_test, 'income')
    print(X_train.shape)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    rfc = RandomForestClassifier(random_state=42)
    rfc = rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)


    # parameter_grid_random = {
    #     'n_estimators': np.arange(10, 2011, 250).tolist(),
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth': [None, 10, 50, 100, 200, 500],
    #     'class_weight': [None, 'balanced', 'balanced_subsample'],
    #     'bootstrap': [True, False]
    # }
    # kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # rfc = RandomForestClassifier(random_state=42, verbose=1)
    # rfc_random = execute_randomized_grid_search(rfc, parameter_grid_random, kf)
    # rfc_random = rfc_random.fit(X_train, y_train)


    # parameter_grid = {
    #     'n_estimators': np.arange(950, 1030, 10).tolist(),
    #     'max_features': ['log2'],
    #     'max_depth': np.arange(6, 21, 2).tolist(),
    #     'bootstrap': [False]
    # }
    # rfc_grid = RandomForestClassifier(random_state=42)
    # gridsearch = GridSearchCV(estimator=rfc_grid, param_grid=parameter_grid,
    #                           scoring='accuracy', cv=kf, verbose=1,
    #                           n_jobs=-1)
    # grid_findings = gridsearch.fit(X_train, y_train)


    # try min_samples_leaf parameter
    best_model = RandomForestClassifier(
        bootstrap=False,
        max_depth=20,
        max_features='log2',
        n_estimators=950,
        random_state=42,
        min_samples_leaf=5
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    show_confusion_matrix(best_model, X_test, y_test, 'confusion matrix after grid search')
    plt.show()
    plot_roc_curve(best_model, X_test, y_test)
    plt.show()


    best_model_balanced = RandomForestClassifier(
        bootstrap=False,
        max_depth=20,
        max_features='log2',
        n_estimators=950,
        random_state=42,
        min_samples_leaf=5,
        class_weight='balanced'
    )
    best_model_balanced.fit(X_train, y_train)
    y_pred = best_model_balanced.predict(X_test)
    print_score(best_model_balanced, X_train, y_train, 'train')
    print_score(best_model_balanced, X_test, y_test, 'test')
    show_confusion_matrix(best_model_balanced, X_test, y_test, 'class_weight=balanced')
    plot_roc_curve(best_model, X_test, y_test)
    plot_roc_curve(best_model_balanced, X_test, y_test)
    plt.show()
