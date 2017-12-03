import os
import math

from sklearn.ensemble import BaggingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import RepeatedKFold, ShuffleSplit
from sklearn.metrics import precision_score
# from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def get_cross_validated_business_data(splits, repeats):
    business_df = pd.read_csv(os.path.join('data', 'biz_csv', 'business.csv'))
    training_data = list()
    test_data = list()
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=0)
    for train, test in rkfold.split(business_df):
        training_data.append(list(business_df.values[train]))
        test_data.append(list(business_df.values[test]))
    return training_data, test_data


def get_train_test_split():
    business_df = pd.read_csv(os.path.join('data', 'biz_csv', 'business.csv'))
    training_data = list()
    test_data = list()
    indices = ShuffleSplit(n_splits=2, test_size=.05, random_state=0)
    for train, test in indices.split(business_df):
        training_data.append(list(business_df.values[train]))
        test_data.append(list(business_df.values[test]))
    return training_data, test_data


def get_regression_model(biz_id_train):
    data = pd.read_csv(os.path.join('data', 'checkin_csv', 'checkin.csv'))
    train = pd.DataFrame()
    i = 1
    while biz_id_train:
        temp = data.loc[data['business_id'] == biz_id_train.pop()]
        i += 1
        train = train.append(temp)
    if len(train) != 0:
        X_train = train[train.columns.values[:-1]].values
        y_train = train[train.columns.values[-1]].values
        # models[location][category] = MLPRegressor(hidden_layer_sizes=(10, 5), activation='relu', learning_rate='adaptive').fit(X_train, y_train)
        return BaggingRegressor().fit(X=X_train, y=y_train)
    return None


def make_regression_models(clustered_business_ids):
    data = pd.read_csv(os.path.join('data', 'checkin_csv', 'checkin.csv'))
    models = dict()
    for location in clustered_business_ids:
        models[location] = dict()
        for category in clustered_business_ids[location]:
            print(location, category)
            biz_id_train = clustered_business_ids[location][category]
            train = pd.DataFrame()
            i = 1
            while biz_id_train:
                temp = data.loc[data['business_id'] == biz_id_train.pop()]
                i += 1
                train = train.append(temp)
            if len(train) != 0:
                X_train = train[train.columns.values[:-1]].values
                y_train = train[train.columns.values[-1]].values
                # models[location][category] = MLPRegressor(hidden_layer_sizes=(10, 5), activation='relu', learning_rate='adaptive').fit(X_train, y_train)
                models[location][category] = BaggingRegressor().fit(X=X_train, y=y_train)
    return models


def test_model(model, biz_id_test):
    if model is not None:
        data = pd.read_csv(os.path.join('data', 'checkin_csv', 'checkin.csv'))
        test = data.loc[data['business_id'] == biz_id_test]
        if len(test) != 0:
            X_test = test[test.columns.values[:-1]].values
            y_test = test[test.columns.values[-1]].values
            predicted_values = model.predict(X_test)
            predicted_values = [0 if x < 0 else math.floor(x) for x in predicted_values]
            # print(predicted_values)
            # print(y_test)
            # print(model.score(X_test, y_test))
            print(precision_score(y_test, predicted_values, average='macro'))

        else:
            print('No check-in data available for this business to test...')
    else:
        print('No sufficient businesses available in the surrounding area and for this category to find the crowd...')


def cluster_by_position_and_category_cv(n_clusters_pos, n_clusters_cat, cv_splits, cv_repeats):
    # biz_train, biz_test = get_cross_validated_business_data(splits=cv_splits, repeats=cv_repeats)
    biz_train, biz_test = get_train_test_split()
    # for cv_index in range(len(biz_train)):  # iterates through all cross-validated training data
    cv_index = 0
    location_model = KMeans(n_clusters=n_clusters_pos, init='k-means++', max_iter=100, n_init=1, random_state=1)
    train_iterator = biz_train[cv_index]  # this variable has 80% train data
    lat_lon_list = [[el[1], el[2]] for el in
                    train_iterator]  # take out the latitude and longitude out of each row of training data
    predictions_for_training_data = location_model.fit_predict(
        lat_lon_list)  # fit the training data and predict the clusters
    cluster_buckets = dict()
    for i in range(n_clusters_pos):
        cluster_buckets[i] = list()
    for index in range(0, len(lat_lon_list)):
        cluster_buckets[predictions_for_training_data[index]].append(
            train_iterator[index])  # put each data to its corresponding cluster bucket

    # now for each bucket, we need to use the data in the bucket and cluster the data based on the category of the business
    categorical_cluster_models = dict()
    categorical_cluster_data = dict()
    for loc_cluster in cluster_buckets:  # cluster has the cluster number.. 0 to n
        data = cluster_buckets[loc_cluster]
        categories = [line[-1] for line in data]
        vectorizer = TfidfVectorizer(stop_words='english')  # ,token_pattern='[a-zA-Z0-9\s&]+'
        # term-frequency x inverse-document frequency
        # tokenize based on comma instead of space. Otherwise, words like 'Public Services' will not be seen as a single word
        # by the vectorizer
        cluster_no = min(len(categories), n_clusters_cat)
        categorical_model = KMeans(n_clusters=cluster_no, init='k-means++', max_iter=100, n_init=1, random_state=1)
        cat_predictions_for_training_data = categorical_model.fit_predict(vectorizer.fit_transform(categories))

        categorical_cluster_data[loc_cluster] = dict()
        for i in range(cluster_no):
            categorical_cluster_data[loc_cluster][i] = list()

        for index in range(len(categories)):
            pred = cat_predictions_for_training_data[index]
            categorical_cluster_data[loc_cluster][pred].append(data[index][0])
        categorical_cluster_models[loc_cluster] = (categorical_model, vectorizer)
        # categorical_cluster_models[cluster] = (cluster_by_category_cv(data=cluster_buckets[cluster], n_clusters=min(len(cluster_buckets[cluster]),n_clusters_cat)))

    del cluster_buckets
    # At the end of the above loop, we will have n_clusters of categorical clustering models.
    # The index of the model gives the locational cluster to which it belongs to...

    # models = make_regression_models(categorical_cluster_data)
    models = dict()
    for i in range(n_clusters_pos):
        models[i] = dict()
        for j in range(n_clusters_cat):
            models[i][j] = None
    print('created models for every cluster')

    test_iterator = biz_test[cv_index]  # this variable has 20% test data
    lat_lon_list = [[el[1], el[2]] for el in test_iterator]
    predictions_for_test_data = location_model.predict(lat_lon_list)
    # clusters = set()
    count = 0
    # print(len(lat_lon_list))
    for index in range(len(lat_lon_list)):
        print(count)
        count += 1
        locational_prediction = predictions_for_test_data[index]
        model = categorical_cluster_models[locational_prediction][0]
        vectorizer = categorical_cluster_models[locational_prediction][1]
        data = test_iterator[index]
        categorical_prediction = model.predict(vectorizer.transform([data[-1]]))[0]
        print(locational_prediction, categorical_prediction)
        # clusters.add((locational_prediction,categorical_prediction))

        if models[locational_prediction][categorical_prediction] is None:
            models[locational_prediction][categorical_prediction] = get_regression_model(
                categorical_cluster_data[locational_prediction][categorical_prediction])

        # train_bid = categorical_cluster_data[locational_prediction][categorical_prediction]
        # test_bid = data[0]

        test_model(models[locational_prediction][categorical_prediction], data[0])

    # print(len(clusters))


cluster_by_position_and_category_cv(60, 60, 10, 1)
