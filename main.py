from typing import Tuple

import dask.dataframe as dd


def read_data() -> Tuple[dd.DataFrame, dd.DataFrame]:
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

    selected_cols = ['City', 'Country_code', 'Workplace_type', 'Experience_level', 'Remote', 'if_permanent', 'if_b2b', 'if_mandate']
    selected_cols_mean = ['permanent_mean', 'b2b_mean', 'mandate_mean']

    return data[selected_cols].values.compute().transpose(), data[selected_cols_mean].values.compute().transpose()


if __name__ == '__main__':
    X, y = read_data()
    print(X)
    print(y)
