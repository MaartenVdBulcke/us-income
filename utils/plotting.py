import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import plot_confusion_matrix


def show_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, title: str):
    plot_confusion_matrix(model, X_test, y_test)
    plt.title(title)
    plt.show()

