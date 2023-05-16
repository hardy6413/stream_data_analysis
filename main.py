import dask.dataframe as dd
import dask.array as da
import matplotlib.pyplot as plt
from dask_ml.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from dask_ml.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import classification_report, precision_score, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, roc_auc_score
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

    data = data[data.salary_to_b2b > 1000]
    data = data[data.salary_from_b2b > 1000]
    #data = data[data.salary_to_permanent > 1000]
    #data = data[data.salary_from_permanent > 1000]

    #data = data[data.salary_to_b2b < 40000]
    #data = data[data.salary_from_b2b < 40000]
    #data = data[data.salary_to_permanent < 40000]
    #data = data[data.salary_from_permanent < 40000]

    data.currency_exchange_rate = data.currency_exchange_rate.replace(0, 1)
    #data['permanent_mean'] = data[['salary_from_permanent', 'salary_to_permanent']].mean(axis=1) * data[
    #    'currency_exchange_rate']
    #data[data['salary_to_permanent'] == 0].salary_to_permanent.value_counts().compute()
    #data = data[data.currency_exchange_rate == 1]
    data = data[(data.Marker_icon  == 'java') | (data.Marker_icon == 'php')
                |(data.Marker_icon == 'python')
                |(data.Marker_icon == 'devops')
                |(data.Marker_icon == 'net')
                |(data.Marker_icon == 'mobile')
                |(data.Marker_icon == 'javascript')
                |(data.Marker_icon == 'analytics')
                |(data.Marker_icon == 'architecture')
                |(data.Marker_icon == 'c')
                |(data.Marker_icon == 'data')
                |(data.Marker_icon == 'testing')
                |(data.Marker_icon == 'ux')
                ]
    data['currency_exchange_rate'] = 1 / data['currency_exchange_rate']
    data['b2b_mean'] = data[['salary_from_b2b', 'salary_to_b2b']].mean(axis=1) * data['currency_exchange_rate']
    data['permanent_mean'] = data[['salary_from_permanent', 'salary_to_permanent']].mean(axis=1) * data['currency_exchange_rate']
    #data['b2b_mean'] = data[['permanent_mean', 'b2b_mean']].mean(axis=1)
    #data = data[data.b2b_mean > 1000]
    data = data[data.b2b_mean < 39000]
    data = data.sort_values('b2b_mean')
    #parted_df = data.repartition(npartitions=45)
    #data = parted_df.partitions[0]
    return data['City'].values.compute(), \
        data['Workplace_type'].values.compute().transpose(), data['Experience_level'].values.compute().transpose(), \
        data['Remote'].values.compute().transpose(), data['if_permanent'].values.compute().transpose(), \
        data['if_b2b'].values.compute().transpose(), data['if_mandate'].values.compute().transpose(),\
        data['b2b_mean'].values.compute().transpose(), data['Marker_icon'].compute().transpose()
    #data['permanent_mean'].values.compute().transpose(),  data['b2b_mean'].values.compute().transpose(), \
    #data['mandate_mean'].values.compute().transpose()


def transform_strings_to_int(frame: dd.DataFrame):
    data = dd.from_array(frame)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(data)
    return encoded


def standardize_values(frame: dd.DataFrame) -> da.array:
    data = da.array(frame.reshape(-1, 1))

    # TODO: choose proper scaler StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    #xscaled = scaler.fit_transform(data)
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    #print("xx")
    #print(scaled_data.compute())
    #print("yy")
    #print(data.compute())
    return scaled_data


if __name__ == '__main__':
    city, workplace, experience, remote, permanent, b2b, mandate, b2b_mean, language = read_data()
    # print(city)

    city_trans = transform_strings_to_int(city)
    workplace_trans = transform_strings_to_int(workplace)
    experience_trans = transform_strings_to_int(experience)
    remote_trans = transform_strings_to_int(remote)
    language_trans = transform_strings_to_int(language)
    #permanent_trans = transform_strings_to_int(permanent)
    #b2b_trans = transform_strings_to_int(b2b)
    #mandate_trans = transform_strings_to_int(mandate)

    # print('label encoded data')
    # print(city_trans)

    city_trans_stand = standardize_values(city_trans)
    workplace_trans_stand = standardize_values(workplace_trans)
    experience_trans_stand = standardize_values(experience_trans)
    remote_trans_stand = standardize_values(remote_trans)
    language_trans_stand = standardize_values(language_trans)
    #permanent_trans_stand = standardize_values(permanent_trans)
    #b2b_trans_stand = standardize_values(b2b_trans)
    #mandate_trans_stand = standardize_values(mandate_trans)

    #permanent_mean_stand = standardize_values(permanent_trans)
    b2b_mean_stand = standardize_values(b2b_mean)
    #mandate_mean_stand = standardize_values(mandate_trans)

    #print('normalized data')
    #print(city_trans_stand.compute())
    #print(mandate_mean_stand.compute())

    #X_train_n, X_test_n, x_train_n, x_test_n = train_test_split(X_n, xscaled.ravel(), test_size=0.2, random_state=0)

    # build model
    #X = da.concatenate((city_trans_stand, mandate_trans_stand, workplace_trans_stand,
     #                  experience_trans_stand, remote_trans_stand, permanent_trans_stand, b2b_trans_stand),  axis=1)

    poly = PolynomialFeatures(degree=1, include_bias=True)
    #w = poly.fit_transform(city_trans_stand)

    X = da.concatenate([city_trans_stand, language_trans_stand, experience_trans_stand, workplace_trans_stand], axis=1)
    #print(X.compute())
    #print(X.compute())
    X = poly.fit_transform(X)

    # # divide model to train and learn data
    X_train, X_test, y_train, y_test = train_test_split(X, b2b_mean_stand, test_size=0.2, random_state=0)#sprawdzic random state
    # print(X_train)
    # # linear regression with multiple params
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #print(classification_report(y_test, y_pred))
    print(
        'mean_squared_error : ', mean_squared_error(y_test, y_pred))
    print(
        'mean_absolute_error : ', mean_absolute_error(y_test, y_pred))

    score = r2_score(y_test, y_pred)
    print("The accuracy of our model is {}%".format(round(score, 2) * 100))
    print("xx")
    #print(y_test.compute())
    #print(y_pred)
    #r = roc_auc_score(y_test, y_pred)
    #print(r)
    #print("The auc of our model is {}%".format(round(r, 2) * 100))
    # compute Accuracy, Loss, AUC, MAE, RMS from sklearn.metrics
    plt.plot(b2b_mean_stand)
    plt.show()
