from typing import Tuple

import dask.dataframe as dd
import dask.array as da
import numpy as np
from dask_ml.preprocessing import StandardScaler
from dask_ml.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from itertools import chain


def read_data():
    data = dd.read_csv('./archive/data.csv',
                       dtype={'Company_size_from': 'object',
                              'Company_size_to': 'object',
                              'salary_from_mandate': 'float64',
                              'salary_to_mandate': 'float64',
                              'City': 'object',
                              'Country_code': 'object',
                              'Experience_level': 'object',
                              'Remote': 'object',
                              'if_permanent': 'object',
                              'if_b2b': 'object',
                              'if_mandate': 'object'
                              })

    data['permanent_mean'] = data[['salary_from_permanent', 'salary_to_permanent']] \
                                 .mean(axis=1) * data['currency_exchange_rate']

    data['b2b_mean'] = data[['salary_from_b2b', 'salary_to_b2b']] \
                           .mean(axis=1) * data['currency_exchange_rate']
    data['mandate_mean'] = data[['salary_from_mandate', 'salary_to_mandate']] \
                               .mean(axis=1) * data['currency_exchange_rate']

    return data['City'].values.compute().transpose(), \
        data['Workplace_type'].values.compute().transpose(), data['Experience_level'].values.compute().transpose(), \
        data['Remote'].values.compute().transpose(), data['if_permanent'].values.compute().transpose(), \
        data['if_b2b'].values.compute().transpose(), data['if_mandate'].values.compute().transpose()
        # data['permanent_mean'].values.compute().transpose(),  data['b2b_mean'].values.compute().transpose(), \
        # data['mandate_mean'].values.compute().transpose()


def transform_strings_to_int(frame: dd.DataFrame):
    data = dd.from_array(frame)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(data)
    return encoded.compute()


def standardize_values(frame: dd.DataFrame) -> dd.DataFrame:
    data = da.array(frame)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    return scaled_data.compute()



if __name__ == '__main__':
    city, workplace, experience, remote, permanent, b2b, mandate = read_data()
    print(city)
    print(mandate)
    city_trans = transform_strings_to_int(city)
    workplace_trans = transform_strings_to_int(workplace)
    experience_trans = transform_strings_to_int(experience)
    remote_trans = transform_strings_to_int(remote)
    permanent_trans = transform_strings_to_int(permanent)
    b2b_trans = transform_strings_to_int(b2b)
    mandate_trans = transform_strings_to_int(mandate)

    print('label encoded data')
    print(city_trans)
    print(workplace_trans)
    print(experience_trans)
    print(remote_trans)
    print(permanent_trans)
    print(b2b_trans)
    print(mandate_trans)

    city_trans_stand = standardize_values(city_trans)
    workplace_trans_stand = standardize_values(workplace_trans)
    experience_trans_stand = standardize_values(experience_trans)
    remote_trans_stand = standardize_values(remote_trans)
    permanent_trans_stand = standardize_values(permanent_trans)
    b2b_trans_stand = standardize_values(b2b_trans)
    mandate_trans_stand = standardize_values(mandate_trans)

    print('normalized data')
    print(city_trans_stand)
    print(workplace_trans_stand)
    print(experience_trans_stand)
    print(remote_trans_stand)
    print(permanent_trans_stand)
    print(b2b_trans_stand)
    print(mandate_trans_stand)
