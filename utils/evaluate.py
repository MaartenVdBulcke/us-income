import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


def print_score(model, X: pd.DataFrame, y: pd.Series, set: str) -> None:
    score = model.score(X, y)
    string_to_print = 'accuracy score on ' + set + ' set:'
    print(string_to_print, score)


def print_roc_auc_score(y_test: pd.Series, y_pred_proba: np.ndarray, model_title: str) -> None:
    string_to_print = 'AUC score for ' + model_title + ':'
    score = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(string_to_print, score)


def execute_randomized_grid_search(model, parameters, kf):
    random_grid = RandomizedSearchCV(estimator=model,
                       param_distributions=parameters,
                       cv=kf,
                       n_iter=20, verbose=1, random_state=42, n_jobs=-1)
    return random_grid

