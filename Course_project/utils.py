import numpy as np
import dask.dataframe as dd
import pandas as pd
from datetime import date
import datetime
import matplotlib.pyplot as plt


def prepare_dataset(data_path, features_path):
    users_data_df = pd.read_csv(data_path)
    users_data_df = users_data_df.drop('Unnamed: 0', axis=1)

    features_df = dd.read_csv(features_path, sep="\t")
    features_df = features_df.drop('Unnamed: 0', axis=1)

    users_data_df['date'] =  users_data_df['buy_time'].apply(lambda x: date.fromtimestamp(x))

    # Для экспериментов возьмем не очень большой набор данных.
    # Также разделим данные по неделям.

    users_data_df['id'].value_counts().sort_values(ascending=False)

    user_ids = users_data_df['id'].unique()

    features_df = features_df.compute()
    features_df = features_df.loc[(features_df['id'].isin(user_ids))]
    # features_df.to_csv('data/features_df.csv', index=False)

    users_data_df = users_data_df.sort_index()

    users_data_df = users_data_df.sort_values(by="buy_time")

    features_df = features_df.sort_values(by="buy_time")

    data_set = pd.merge_asof(users_data_df, features_df, on='buy_time', by='id')


    data_set['week_on_month'] = data_set['date'].apply(lambda x: pd.to_datetime(x).day//7)
    data_set['day'] = data_set['date'].apply(lambda x: pd.to_datetime(x).day)
    data_set['month'] = data_set['date'].apply(lambda x: pd.to_datetime(x).month)
    # data_set['year'] = data_set['date'].apply(lambda x: pd.to_datetime(x).year)
    data_set = data_set.drop('date', axis=1)
    data_set = data_set.drop('buy_time', axis=1)

    # data_test.to_csv('data/data_test_new.csv', index=False)
    return data_set
