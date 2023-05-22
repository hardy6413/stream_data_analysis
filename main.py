import sys
import time
import schedule
import dask.dataframe as dd
import dask.array as da
from dask_ml.preprocessing import MinMaxScaler
from dask_ml.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
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
                              'if_mandate': 'object',
                              'Marker_icon': 'object'
                              })

    # Filtrowanie danych na podstawie pewnych kryteriów
    data = data[data.salary_to_b2b > 1000]
    data = data[data.salary_from_b2b > 1000]
    data = data[data.salary_to_permanent > 1000]
    data = data[data.salary_from_permanent > 1000]

    # Zamiana zerowych wartości na 1 w kolumnie "currency_exchange_rate"
    data.currency_exchange_rate = data.currency_exchange_rate.replace(0, 1)
    
    # Filtrowanie danych na podstawie wartości w kolumnie "Marker_icon"
    data = data[(data.Marker_icon  == 'java') | (data.Marker_icon == 'php')
                | (data.Marker_icon == 'python')
                | (data.Marker_icon == 'devops')
                | (data.Marker_icon == 'net')
                | (data.Marker_icon == 'mobile')
                | (data.Marker_icon == 'javascript')
                | (data.Marker_icon == 'data')
                | (data.Marker_icon == 'testing')
                ]
    # Obliczanie nowych kolumn i filtrowanie danych
    data['currency_exchange_rate'] = 1 / data['currency_exchange_rate']
    data['b2b_mean'] = data[['salary_from_b2b', 'salary_to_b2b']].mean(axis=1) * data['currency_exchange_rate']
    data['permanent_mean'] = data[['salary_from_permanent', 'salary_to_permanent']].mean(axis=1) * data['currency_exchange_rate']
    data = data[data.b2b_mean > 1000]
    data = data[data.b2b_mean < 39000]

    # Zwracanie wartości z wybranych kolumn
    return data['City'].values.compute(), \
           data['Workplace_type'].values.compute().transpose(), \
           data['Experience_level'].values.compute().transpose(), \
           data['Remote'].values.compute().transpose(), \
           data['if_permanent'].values.compute().transpose(), \
           data['if_b2b'].values.compute().transpose(), \
           data['if_mandate'].values.compute().transpose(), \
           data['b2b_mean'].values.compute().transpose(), \
           data['Marker_icon'].compute().transpose()


def transform_strings_to_int(frame: dd.DataFrame):
    # Konwersja wartości typu string na liczby za pomocą LabelEncoder
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(frame)
    return encoded


def standardize_values(frame: dd.DataFrame) -> da.array:
    # Standaryzacja wartości przy użyciu MinMaxScaler    
    data = da.array(frame.reshape(-1, 1))
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data


def update_data():
    # Aktualizacja danych
    city, workplace, experience, remote, permanent, b2b, mandate, b2b_mean, language = read_data()

    city_trans = transform_strings_to_int(city)
    workplace_trans = transform_strings_to_int(workplace)
    experience_trans = transform_strings_to_int(experience)
    remote_trans = transform_strings_to_int(remote)
    language_trans = transform_strings_to_int(language)

    city_trans_stand = standardize_values(city_trans)
    workplace_trans_stand = standardize_values(workplace_trans)
    experience_trans_stand = standardize_values(experience_trans)
    remote_trans_stand = standardize_values(remote_trans)
    language_trans_stand = standardize_values(language_trans)

    b2b_mean_stand = standardize_values(b2b_mean)

    poly = PolynomialFeatures(degree=8, include_bias=True)

    X = da.concatenate([city_trans_stand, language_trans_stand, experience_trans_stand], axis=1)

    X = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, b2b_mean_stand, test_size=0.1, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    sys.stdout.write('mean_squared_error: {}\n'.format(mse))
    sys.stdout.write('mean_absolute_error: {}\n'.format(mae))
    sys.stdout.write("The accuracy of our model is {}%\n".format(round(r2, 2) * 100))
    sys.stdout.write("-----------------------------------------\n")
    sys.stdout.flush()

if __name__ == '__main__':
    # Uruchamianie aktualizacji danych co godzinę
    schedule.every().minute.do(update_data)

    while True:
        # Wykonywanie zaplanowanych zadań
        schedule.run_pending()
        time.sleep(1)
