"""Preprocessing modifies the structure of the data so the train_model.py can read the data properly

Its split into 2 steps to avoid data leakages:

fit_preprocessing() - learns the imputation values, encoded column layout and scales the statistics
from the training set only
preprocessing_data() - applies those learnt rules to any of the datasets so all of them are transformed
identicaly
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_column_names(df):
    """ Strips away the whiteapce from column names.

    Args: df: raw dataframe
    Return: a copy of df with cleaned column names so the input isnt modified
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def fit_preprocessing(df_train, target_column):
    """ Learns all preprocessing rules from the traning data.

    Args: df_train: training dataframe
          target_column: label column,'Match_ready'
    Returns: dictionary of learnt rules to be passed to preprocessing_data
            - numeric_cols: names of numeric feature columns
            - categorical_cols: names of categorical feature columns
            - medians: per-column median, used to fill missing values
            - encoded_cols: final column order after one-hot encoding
            - scaler: fitted StandardScaler() for the numeric columns
    """

    train_rules = df_train.drop(columns = [target_column])

    numeric_cols = train_rules.select_dtypes(include="number").columns.tolist()
    categorical_cols = train_rules.select_dtypes(exclude="number").columns.tolist()

    # Imputation
    
    medians = train_rules[numeric_cols].median()

    # Encoding categorical attributes (One Hot Encoding)

    train_features_encoded = pd.get_dummies(train_rules, columns = categorical_cols, drop_first = True)
    encoded_cols = train_features_encoded.columns.to_list()

    # Scaling(standardization)

    scaler = StandardScaler()
    scaler.fit(train_features_encoded[numeric_cols])

    return {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "medians": medians, "encoded_cols": encoded_cols, "scaler": scaler}

def preprocessing_data(df,preprocessing_rules , target_column = None):
    """Applying 'rules' learnt from fit_processing() to a dataframe.

    Args: df: dataframe to process
          preprocessing_rules: dictionary retuned by fit_prepreocessing
          target_column: the label column name 'Match_Ready', if 'None' no labels are returned
    Returns: tuple (X,y): the processed feature matrix and the labels 
    """

    df = df.copy()

    if target_column is not None and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns = [target_column])
    else:
        y = None
        X = df

    # Imputation
    # Fill gaps with trainings medians (not this dataset's), to avoid leakage

    X[preprocessing_rules["numeric_cols"]] = X[preprocessing_rules["numeric_cols"]].fillna(preprocessing_rules["medians"])

    # One hot encoding

    X = pd.get_dummies(X, columns = preprocessing_rules["categorical_cols"], drop_first = True)

    # Column alignment
    # Force the same columns in the same order, filling gaps with 0

    X= X.reindex(columns = preprocessing_rules["encoded_cols"], fill_value = 0)  # column alignment
    X = X.astype(float) #so numeric value are retuned(not bool)

    # Scaling
    scaler = preprocessing_rules["scaler"]
    numeric_cols = preprocessing_rules["numeric_cols"]

    X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X,y
