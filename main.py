import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report



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

rfc = RandomForestClassifier()
rfc = rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(rfc, X_test, y_test)
plt.title('Consufion matrix default Random Forest')
plt.show()