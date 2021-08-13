import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from utils.determine_base_accuracy_score import *
from utils.dataset_manipulation import *
from utils.plotting import *
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

income_train = pd.read_csv(r'datasets\US Income\cleaned\data_train.csv')
income_test = pd.read_csv(r'datasets\US Income\cleaned\data_test.csv')

if __name__=='__main__':
    X_train, y_train = split_features_and_target(income_train, 'income')
    X_test, y_test = split_features_and_target(income_test, 'income')

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)


    # base accuracy score: 0.85043916...
    rfc = RandomForestClassifier(random_state=42)
    rfc = rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    print_score(rfc, X_train, y_train, 'train')
    print_score(rfc, X_test, y_test, 'test')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    show_confusion_matrix(rfc, X_test, y_test, 'Confusion matrix default Random Forest')
    plot_roc_curve(rfc, X_test, y_test)
    plt.show()

    y_pred_proba = rfc.predict_proba(X_test)
    print_roc_auc_score(y_test, y_pred_proba, 'default model')


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
    plt.ylim([0,1])
    plt.title('evolution of f1 scores over different models')
    plt.xlabel('model')
    plt.ylabel('f1 scores')
    plt.legend()
    plt.show()

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
    plt.show()

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
    plt.show()


    # first improvement: more complex accuracy measurement: cross validation
    # combine train and test dataset
    income_combined = pd.concat([income_train, income_test], axis=0)
    print(income_combined.head())
    print(income_combined.shape)
    X = income_combined.drop(['income'], axis=1)
    y = income_combined['income']
    scores = cross_val_score(rfc, X, y, cv=10) # cv is the number of folds (k)
    print(scores)

    # It is always a good practice to show the mean AND the standard deviation of the model accuracy
    print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))

    ###############################
    ###############################

