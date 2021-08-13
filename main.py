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

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

income_train = pd.read_csv(r'datasets\US Income\cleaned\data_train.csv')
income_test = pd.read_csv(r'datasets\US Income\cleaned\data_test.csv')

print(income_train.shape)
print(income_train.head())
print(income_train.shape)
print(income_test.head())
print(income_train.isnull().sum().sum())
print(income_test.isnull().sum().sum())
print(income_train.income.value_counts())
print(income_test.income.value_counts())

categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'native-country']
    # , 'income']
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
             'hours-per-week']

# split features and target
X_train = income_train.drop(['income'], axis=1)
y_train = income_train['income']
X_test = income_test.drop(['income'], axis=1)
y_test = income_test['income']

# base accuracy score: 0.85043916...

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

rfc = RandomForestClassifier(random_state=42)
rfc = rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(rfc.score(X_test, y_test))
print('score on train', rfc.score(X_train, y_train))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(rfc, X_test, y_test)
plt.title('Confusion matrix default Random Forest')
plt.show()

plot_roc_curve(rfc, X_test, y_test)
plt.show()

y_pred_proba = rfc.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred_proba[:, 1]))

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
