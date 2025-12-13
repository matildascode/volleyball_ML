import pandas as pd
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("player_data.csv")
print(df_train.head())

def fit_preprocessing(df_test, target_column):

    #imputation

    columns_with_empty_elements = df_train.columns[df_train.isna().any()].tolist()
    for i in range(0,len(columns_with_empty_elements)):
        current_column =columns_with_empty_elements[i]
        column_median = df_train[current_column].median()

        df_train[current_column] = df_train[current_column].fillna(column_median)

    #encoding categorical attributes (One Hot Encoding)

    non_numerical_columns = [col for col in df_train.columns: if not pd.api.types.is_numeric_dtype(df_train[col])]
    for i in range(0,len(non_numerical_columns)):
        dummies =  pd.get_dummies(df_train[non_numerical_columns[i]], drop_first = True)

    #scalling(standardization)

    total_rows = len(df_train)
    THRESHOLD = 0.05

    scaling_columns = [col for col in df_train.columns: if df_train[col].nunique()/total_rows > THRESHOLD]
    scaler = StandardScaler()
    scaler.fit(df_train[scaling_cols])

    df_train[scaling_cols] = scaler.transform(df_train[scaling_cols])









