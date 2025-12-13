import pandas as pd
from sklearn.preprocessing import StandardScaler

def fit_preprocessing(df_train, target_column):
    #Learning how to process data

    train_rules = df_train.drop(columns = [target_column])

    numeric_cols = train_rules.select_dtypes(include="number").columns.tolist()
    categorical_cols = train_rules.select_dtypes(exclude="number").columns.tolist()

    # Imputation
    
    medians = train_rules[numeric_cols].median()

    # Encoding categorical attributes (One Hot Encoding)

    train_features_encoded = pd.get_dummies(train_rules, columns = categorical_cols, drop_first = True)
    encoded_cols = train_features_encoded.columns.to_list()

    # Scalling(standardization)

    scaler = StandardScaler()
    scaler.fit(train_features_encoded[numeric_cols])

    return{"numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "medians": medians, "encoded_cols": encoded_cols, "scaler": scaler}

def preprocessing_data(df,preprocessing_rules , target_column = None):
    #Applying 'rules' learnt from fit_processing function

    df = df.copy()

    if target_column != None and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns = [target_column])
    else:
        y = None
        X = df

    # Imputation

    X[preprocessing_rules["numeric_cols"]] = X[preprocessing_rules["numeric_cols"]].fillna(preprocessing_rules["medians"])

    # One hot encoding

    X = pd.get_dummies(X, columns = preprocessing_rules["categorical_cols"], drop_first = True)

    X= X.reindex(columns = preprocessing_rules["encoded_cols"], fill_value = 0)  # column alingment
    X = X.astype(float) #so numeric value are retuned(not bool)

    # Scalling
    scaler = preprocessing_rules["scaler"]
    numeric_cols = preprocessing_rules["numeric_cols"]

    X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X,y

df_train = pd.read_csv("training_set.csv")
df_val = pd.read_csv("validation_set.csv")

preprocessing_rules = fit_preprocessing(df_train, "Match_Ready")
print(preprocessing_data(df_val, preprocessing_rules, "Match_Ready"))