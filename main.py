import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    train_test_split, KFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE

from utils.determine_base_accuracy_score import *
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

    print_score(rfc, X_train, y_train, 'train')
    print_score(rfc, X_test, y_test, 'test')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # show_confusion_matrix(rfc, X_test, y_test, 'Confusion matrix default Random Forest')
    plot_roc_curve(rfc, X_test, y_test)
    # plt.show()
    y_pred_proba = rfc.predict_proba(X_test)
    print_roc_auc_score(y_test, y_pred_proba, 'default model')
    print(roc_auc_score(y_test, y_pred_proba[:,1], average='weighted'))




    ######### PLOTTING THIS TO PUT IN FUNCTIONS #########
    # dict with model number + [f1_score_0, f1_score_1]
    f1_score_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_score_1 = f1_score(y_test, y_pred, pos_label=1)
    models = [0]
    f1_scores_0 = [f1_score_0]
    f1_scores_1 = [f1_score_1]
    train_scores = []
    train_scores.append(rfc.score(X_train, y_train))
    test_scores = []
    test_scores.append(accuracy_score(y_test, y_pred))

    plt.plot(models, f1_scores_0, label='f1 score <= 50k', color='blue', markersize=5, marker='o')
    plt.plot(models, f1_scores_1, label='f1 score > 50k', color='orange', markersize=5, marker='o')
    plt.xticks([0, 1, 2, 3])
    plt.xlim([-0.5, 3.5])
    plt.ylim([0, 1])
    plt.title('evolution of f1 scores over different models')
    plt.xlabel('model')
    plt.ylabel('f1 scores')
    plt.legend()
    # plt.show()

    # plt.plot(models, train_scores, label='accuracy score train set', color='red', markersize=5, marker='o')
    plt.bar(models, train_scores, label='accuracy score train set', color='red', )
    # plt.plot(models, test_scores, label='accuracy score test set', color='green', markersize=5, marker='o')
    plt.bar(models, test_scores, label='accuracy score test set', color='green', )
    plt.xticks([0, 1, 2, 3])
    plt.xlim([-0.5, 3.5])
    plt.ylim([0, 1.3])
    plt.title('evolution of train and test scores over different models')
    plt.xlabel('model')
    plt.ylabel('accuracy scores')
    plt.legend()
    # plt.show()

    split = ['train', 'test']
    scores = [(rfc.score(X_train, y_train)), rfc.score(X_test, y_test)]
    sns.barplot(x=split, y=scores)
    # plt.xticks([0, 1, 2, 3])
    # plt.xlim([-0.5, 3.5])
    plt.ylim([0, 1.2])
    # plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.title('comparison of train and test scores for default model')
    # plt.xlabel('')
    plt.ylabel('accuracy score')
    # plt.legend()
    # plt.show()

    ############
    # STRATIFY # NO BIG RESULTS
    ############
    # income_all = pd.concat([income_test,income_train])
    # print(income_all.head())
    # print(income_all.shape)

    # all_X = income_all.drop('income', axis=1)
    # all_y = income_all.income

    # X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2,
    #                                                     stratify=all_y, random_state=42)

    # rfc_all = RandomForestClassifier(random_state=42)
    # rfc_all = rfc_all.fit(X_train, y_train)
    # y_pred = rfc.predict(X_test)

    # print_score(rfc_all, X_train, y_train, 'train')
    # print_score(rfc_all, X_test, y_test, 'test')
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    #### FIRST IMPROVEMENT ##

    # first improvement: more complex accuracy measurement: cross validation
    # combine train and test dataset
    # income_combined = pd.concat([income_train, income_test], axis=0)
    # X = income_combined.drop(['income'], axis=1)
    # y = income_combined['income']
    # scores = cross_val_score(rfc, X, y, cv=10) # cv is the number of folds (k)
    # scores = cross_val_score(rfc, X_train, y_train, cv=10)  # cv is the number of folds (k)
    # print(scores)
    # print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))

    #################
    # CLASS WEIGHT  #
    #################
    # rfc_class = RandomForestClassifier(random_state=42, class_weight='balanced')
    # rfc_class = RandomForestClassifier(random_state=42, class_weight={0:1, 1:4})
    # rfc_class = rfc_class.fit(X_train, y_train)
    # y_pred_class = rfc_class.predict(X_test)
    # print('CLASS WEIGHT')
    # print_score(rfc_class, X_train, y_train, 'train')
    # print_score(rfc_class, X_test, y_test, 'test')
    # print(confusion_matrix(y_test, y_pred_class))
    # print(classification_report(y_test, y_pred_class))

    # scores = cross_val_score(rfc_class, X_train, y_train, cv=10)  # cv is the number of folds (k)
    # print(scores)
    # print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))

    ###############################
    # OVERSAMPLING #
    ################

    # print(y_train.shape)
    # print('apply class weight')

    # print('start oversampling')
    # X_train_two, X_val, y_train_two, y_val = train_test_split(X_train, y_train, test_size=0.15,
    #                                                           random_state=42, stratify=y_train)

    # oversample = SMOTE(random_state=42, sampling_strategy='all', k_neighbors=25)
    # X_train_res, y_train_res = oversample.fit_resample(X_train_two, y_train_two)
    # print('stop oversampling')
    # print(X_train_res.shape, y_train_res.shape)
    # print(y_train_res.value_counts())

    # rfc_res = RandomForestClassifier(random_state=42)
    # rfc_res = rfc_res.fit(X_train_res, y_train_res)
    # y_pred_res = rfc_res.predict(X_test)

    # print_score(rfc_res, X_train_res, y_train_res, 'train')
    # print_score(rfc_res, X_val, y_val, 'validation')
    # print_score(rfc_res, X_test, y_test, 'test')
    # print(confusion_matrix(y_test, y_pred_res))
    # print(classification_report(y_test, y_pred_res))

    # scores_res = cross_val_score(rfc_res, X_train_res, y_train_res, cv=10)  # cv is the number of folds (k)
    # print(scores_res)
    # print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores_res.mean() * 100, scores_res.std() * 100))

    ###############################

    parameter_grid = {
        'n_estimators': np.arange(10, 2011, 250).tolist(),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 50, 100, 200, 500],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'bootstrap': [True, False]
    }
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    rfc = RandomForestClassifier(random_state=42, verbose=1)
    # print(rfc.get_params().keys())
    rfc_random = RandomizedSearchCV(estimator=rfc,
                                    param_distributions=parameter_grid, n_iter=50,
                                    cv=kf, verbose=1, random_state=42, n_jobs=-1)
    # rfc_random = rfc_random.fit(X_train, y_train)

    # print(rfc_random.best_score_)
    # print(rfc_random.best_estimator_)
    # print(rfc_random.best_params_)
    # RESULTS
    # Fitting 4 folds for each of 50 candidates, totalling 200 fits
    # 0.8598631551340064
    # RandomForestClassifier(bootstrap=False, max_depth=10, max_features='log2',
    #                        n_estimators=1010, random_state=42)
    # {'n_estimators': 1010, 'max_features': 'log2', 'max_depth': 10, 'class_weight': None, 'bootstrap': False}

    ######################
    # GRID SEARCH BASED ON RANDOMGRIDSEARCH
    ######################

    # parameters = {
    #     'n_estimators': np.arange(980, 1030, 9).tolist(),
    #     'max_features': ['log2'],
    #     'max_depth': np.arange(1, 20, 5).tolist(),
    #     # 'class_weight': [None, {0: 1, 1: 4}, {0:1, 1:5}],
    #     'bootstrap': [False]
    # }
    #
    # kf = KFold(n_splits=5, random_state=42, shuffle=True)
    #
    # rfc_grid = RandomForestClassifier(random_state=42)
    # gridsearch = GridSearchCV(estimator=rfc_grid,
    #                           param_grid=parameters,
    #                           scoring='accuracy',
    #                           cv=kf,
    #                           verbose=1,
    #                           n_jobs=-1  # Use all but one CPU core
    #                           )
    #
    # grid_findings = gridsearch.fit(X_train, y_train)
    # print('best parameters: ', grid_findings.best_params_)
    # print('best accuracy: ', (grid_findings.best_score_ * 100))
    # print('best estimator:', grid_findings.best_estimator_)
    ##Fitting 5 folds for each of 24 candidates, totalling 120 fits
    # best parameters:  {'bootstrap': False, 'max_depth': 16, 'max_features': 'log2', 'n_estimators': 980}
    # best accuracy:  86.46539136808597

    ##### RUN AGAIN BASED ON GRID1 RESULTS
    parameters = {
        'n_estimators': np.arange(950, 1010, 11).tolist(),
        'max_features': ['log2'],
        'max_depth': np.arange(10, 21, 2).tolist(),
        # 'class_weight': [None, {0: 1, 1: 4}, {0:1, 1:5}],
        'bootstrap': [False]
    }

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    # rfc_grid = RandomForestClassifier(random_state=42)
    # gridsearch = GridSearchCV(estimator=rfc_grid,
    #                           param_grid=parameters,
    #                           scoring='accuracy',
    #                           cv=kf,
    #                           verbose=1,
    #                           n_jobs=-1  # Use all but one CPU core
    #                           )
    #
    # grid_findings = gridsearch.fit(X_train, y_train)
    # print('best parameters: ', grid_findings.best_params_)
    # print('best accuracy: ', (grid_findings.best_score_ * 100))
    ## RESULTS
    ## Fitting 5 folds for each of 36 candidates, totalling 180 fits
    # best parameters:  {'bootstrap': False, 'max_depth': 18, 'max_features': 'log2', 'n_estimators': 950}
    # best accuracy:  86.48688731024062

    ###### GRID AGAIN #####
    param = {
        'bootstrap': [False],
        'max_depth': [18],
        'max_features': ['log2', 6, 8],
        'n_estimators': [950],
        'min_samples_leaf': [6, 8, 10, 12],  # added
        'criterion': ['gini', 'entropy']

    }
    rfc_grid = RandomForestClassifier(random_state=42)
    gridsearch = GridSearchCV(estimator=rfc_grid,
                              param_grid=param,
                              scoring='accuracy',
                              cv=kf,
                              verbose=1,
                              n_jobs=-1  # Use all but one CPU core
                              )

    grid_findings = gridsearch.fit(X_train, y_train)
    print('best parameters: ', grid_findings.best_params_)
    print('best accuracy: ', (grid_findings.best_score_ * 100))
    best_model = grid_findings.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print('best model')
    print_score(best_model, X_train, y_train, 'train')
    print_score(best_model, X_test, y_test, 'test')
    scores = cross_val_score(best_model, X_train, y_train, cv=kf)
    print(scores)
    print(scores.mean(), scores.std())

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    # best_model = grid_findings.best_estimator_
    best_model = RandomForestClassifier(
        bootstrap=False,
        max_depth=18,
        max_features=4,
        n_estimators=950,
        random_state=42,
        min_samples_leaf=5  # added
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print('best model')
    print_score(best_model, X_train, y_train, 'train')
    print_score(best_model, X_test, y_test, 'test')
    scores = cross_val_score(best_model, X_train, y_train, cv=kf)
    print(scores)
    print(scores.mean(), scores.std())
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # accuracy score on train set: 0.9019993243450753
    # accuracy score on test set: 0.8590381426202321
    # 0.8652681737262575
    # 0.0031096534116204478
    # [[11719   716]
    #  [1579  2267]]

    best_model = RandomForestClassifier(
        bootstrap=False,
        max_depth=18,
        max_features=4,
        n_estimators=950,
        random_state=42,
        min_samples_leaf=5,  # added
        class_weight='balanced'
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print('best model')
    print_score(best_model, X_train, y_train, 'train')
    print_score(best_model, X_test, y_test, 'test')
    scores = cross_val_score(best_model, X_train, y_train, cv=kf)
    print(scores)
    print(scores.mean(), scores.std())
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    best_model = RandomForestClassifier(
        bootstrap=False,
        max_depth=18,
        max_features=4,
        n_estimators=950,
        random_state=42,
        min_samples_leaf=5,  # added
        class_weight='balanced_subsample'
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print('best model')
    print_score(best_model, X_train, y_train, 'train')
    print_score(best_model, X_test, y_test, 'test')
    scores = cross_val_score(best_model, X_train, y_train, cv=kf)
    print(scores)
    print(scores.mean(), scores.std())
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    ###### GRID AGAIN #####
    param = {
        'bootstrap': [False],
        'max_depth': [18],
        'max_features': ['log2', 6, 8],
        'n_estimators': [950],
        'min_samples_leaf': [6, 8, 10, 12]  # added
    }
    rfc_grid = RandomForestClassifier(random_state=42)
    gridsearch = GridSearchCV(estimator=rfc_grid,
                              param_grid=param,
                              scoring='accuracy',
                              cv=kf,
                              verbose=1,
                              n_jobs=-1  # Use all but one CPU core
                              )

    grid_findings = gridsearch.fit(X_train, y_train)
    print('best parameters: ', grid_findings.best_params_)
    print('best accuracy: ', (grid_findings.best_score_ * 100))
    best_model = grid_findings.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print('best model')
    print_score(best_model, X_train, y_train, 'train')
    print_score(best_model, X_test, y_test, 'test')
    scores = cross_val_score(best_model, X_train, y_train, cv=kf)
    print(scores)
    print(scores.mean(), scores.std())

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Fitting 5 folds for each of 12 candidates, totalling 60 fits
    # best parameters:  {'bootstrap': False, 'max_depth': 18, 'max_features': 6, 'min_samples_leaf': 6, 'n_estimators': 950}
    # best accuracy:  86.4531115429319
    # best model
    # accuracy score on train set: 0.9038420195939928
    # accuracy score on test set: 0.8584239297340458

