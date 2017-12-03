import os
import random
import warnings

import math
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def test_accuracy(biz_id_train, biz_id_test):
    data = pd.read_csv(os.path.join('data', 'checkin_csv', 'checkin.csv'))
    train = pd.DataFrame()
    print(biz_id_train)
    while biz_id_train:
        temp = data.loc[data['business_id'] == biz_id_train.pop()]
        train = train.append(temp)

    X_train = train[train.columns.values[:-1]].values
    y_train = train[train.columns.values[-1]].values
    models = get_regressors()
    for c in models:
        print(c)
        regr = c.fit(X_train, y_train)
        test = data.loc[data['business_id'] == biz_id_test]
        X_test = test[test.columns.values[:-1]].values
        y_test = test[test.columns.values[-1]].values
        predicted_values = regr.predict(X_test)
        print(predicted_values)
        print(y_test)
        for (x, x_) in zip(y_test, predicted_values):
            print(math.fabs(x - x_) / 100)
        print(regr.score(X_test, y_test))


def get_regressors():
    classifiers = [
        DecisionTreeRegressor(),
        SGDRegressor(),
        MLPRegressor(),
        SVR(),
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
        BaggingRegressor()
    ]
    return classifiers


if __name__ == '__main__':
    data = pd.read_csv(os.path.join('data', 'checkin_csv', 'checkin.csv'))
    data = data.sort_values(by='business_id')
    train_id = random.sample(set(data['business_id'].values), 1000)
    test_id = random.sample(train_id, 1)
    test_ID = test_id.pop()
    test_accuracy(train_id, test_ID)
