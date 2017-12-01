import os

import pandas as pd
from sklearn.neural_network import MLPRegressor


def preprocess_data(df):
    biz_id = set(df['business_id'].values)
    print(biz_id)


def test_accuracy(biz_id_train, biz_id_test):
    data = pd.read_csv(os.path.join('data', 'checkin_csv', 'checkin.csv'))
    train = pd.DataFrame()
    while biz_id_train:
        temp = data.loc[data['business_id'] == biz_id_train.pop()]
        train = train.append(temp)
    # print(train)

    X_train = train[train.columns.values[:-1]].values
    y_train = train[train.columns.values[-1]].values
    # print(X_train)
    # print(y_train)
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(10,5), activation='relu', learning_rate='adaptive').fit(
        X_train,
        y_train)
    test = data.loc[data['business_id'] == biz_id_test]
    X_test = test[test.columns.values[:-1]].values
    y_test = test[test.columns.values[-1]].values
    predicted_values = mlp_regressor.predict(X_test)
    print(predicted_values)
    print(y_test)
    print(mlp_regressor.score(X_test, y_test))


if __name__ == '__main__':
    biz_list = list(range(100))
    biz_test = 10
    test_accuracy(biz_list, biz_test)
