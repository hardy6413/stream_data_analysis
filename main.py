
import dask.dataframe as dd


if __name__ == '__main__':
    df = dd.read_csv('./data.csv', dtype={"Company_size_from": "string", "Company_size_to": "string", "salary_from_mandate": "float", "salary_to_mandate": "float"})
    df.head()
    print(df[['Company_size_from','Company_size_to']].head())