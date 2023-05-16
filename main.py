import dask.dataframe as dd
import dask.array as da

from dask_ml.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from dask_ml.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MaxAbsScaler


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

    data['permanent_mean'] = data[['salary_from_permanent', 'salary_to_permanent']].mean(axis=1) * data[
        'currency_exchange_rate']

    data['b2b_mean'] = data[['salary_from_b2b', 'salary_to_b2b']].mean(axis=1) * data[
        'currency_exchange_rate']

    data['mandate_mean'] = data[['salary_from_mandate', 'salary_to_mandate']].mean(axis=1) * data[
        'currency_exchange_rate']

    # marker_icons = ['java', 'php', 'python', 'devops', 'net', 'mobile', 'javascript',
    #                 'analytics', 'architecture', 'c', 'data', 'testing', 'ux']
    #
    # data = data[data.Marker_icon.isin(marker_icons)]
    # print(data['Company_size_to'].values.compute())
    data = data[data.salary_to_b2b > 1000]
    data = data[data.salary_from_b2b > 1000]

    return data['City'].values.compute().transpose(), \
        data['Workplace_type'].values.compute().transpose(), data['Experience_level'].values.compute().transpose(), \
        data['Remote'].values.compute().transpose(), data['if_permanent'].values.compute().transpose(), \
        data['if_b2b'].values.compute().transpose(), data['if_mandate'].values.compute().transpose(), \
        data['permanent_mean'].values.compute().transpose(), data['b2b_mean'].values.compute().transpose(), \
        data['mandate_mean'].values.compute().transpose(), data['Marker_icon'].values.compute().transpose(), \
        data['Title'].values.compute().transpose()


def transform_strings_to_int(frame: dd.DataFrame):
    data = dd.from_array(frame)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(data)
    return encoded


def standardize_values(frame: dd.DataFrame) -> da.array:
    data = da.array(frame.reshape(-1, 1))

    # TODO: choose proper scaler StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    return scaled_data


def show_results(test, pred):
    print('Mean Squared Error (MSE): {:.4f}'.format(mean_squared_error(test, pred)))
    print('Mean Absolute Error (MAE): {:.4f}'.format(mean_absolute_error(test, pred)))
    print('R^2 Score: {:.4f}'.format(r2_score(test, pred)))


if __name__ == '__main__':
    city, workplace, experience, remote, permanent, b2b, mandate, \
        permanent_mean, b2b_mean, mandate_mean, language, title = read_data()

    city_trans = transform_strings_to_int(city)
    workplace_trans = transform_strings_to_int(workplace)
    experience_trans = transform_strings_to_int(experience)
    remote_trans = transform_strings_to_int(remote)
    permanent_trans = transform_strings_to_int(permanent)
    b2b_trans = transform_strings_to_int(b2b)
    mandate_trans = transform_strings_to_int(mandate)
    language_trans = transform_strings_to_int(language)
    title_trans = transform_strings_to_int(title)

    city_trans_stand = standardize_values(city_trans)
    workplace_trans_stand = standardize_values(workplace_trans)
    experience_trans_stand = standardize_values(experience_trans)
    remote_trans_stand = standardize_values(remote_trans)
    permanent_trans_stand = standardize_values(permanent_trans)
    b2b_trans_stand = standardize_values(b2b_trans)
    mandate_trans_stand = standardize_values(mandate_trans)
    language_trans_stand = standardize_values(language_trans)
    title_trans_stand = standardize_values(title_trans)

    permanent_mean_stand = standardize_values(permanent_mean)
    b2b_mean_stand = standardize_values(b2b_mean)
    mandate_mean_stand = standardize_values(mandate_mean)

    X = da.concatenate([city_trans_stand, experience_trans_stand, workplace_trans_stand, remote_trans_stand,
                        permanent_trans_stand, b2b_trans_stand, mandate_trans_stand, language_trans_stand,
                        title_trans_stand], axis=1)

    poly = PolynomialFeatures(degree=3, include_bias=True)
    poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, b2b_mean_stand, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    show_results(y_test, y_pred)
