import pandas as pd

categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'native-country']
    # , 'income']
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
             'hours-per-week']


def split_features_and_target(df: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series):
    X_df = df.drop([target], axis=1)
    y_df = df[target]
    return X_df, y_df



