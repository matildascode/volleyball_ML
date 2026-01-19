from preprocessing import clean_column_names, fit_preprocessing, preprocessing_data
import pandas as pd
from train_model import MyLogisticRegression
from interactive_AI import model_advice

def main():
    df_train = clean_column_names(pd.read_csv("data_sets/training_set.csv"))
    df_val   = clean_column_names(pd.read_csv("data_sets/validation_set.csv"))
    df_test  = clean_column_names(pd.read_csv("data_sets/test_set.csv"))

    target_column = "Match_Ready"

    preprocessing_rules = fit_preprocessing(df_train, target_column)

    X_train, y_train = preprocessing_data(df_train, preprocessing_rules, target_column)
    X_val, y_val     = preprocessing_data(df_val, preprocessing_rules, target_column)
    X_test, y_test   = preprocessing_data(df_test, preprocessing_rules, target_column)

    model = MyLogisticRegression(monitor_training=True)
    model.fit(X_train.values, y_train.values)
    print("\n✅ Training complete!\n")

    print("Enter player information to predict match readiness:")
    player_input = {}
    for col in df_train.drop(columns=[target_column]).columns:
        value = input(f"{col}: ")
        try:
            value = float(value)
        except ValueError:
            pass 
        player_input[col] = value

    model_advice(player_input, model, preprocessing_rules)

from preprocessing import clean_column_names, fit_preprocessing, preprocessing_data
import pandas as pd
from train_model import MyLogisticRegression

def test_model():
    # -----------------------
    # 1. Load datasets
    # -----------------------
    df_train = clean_column_names(pd.read_csv("data_sets/training_set.csv"))
    df_val   = clean_column_names(pd.read_csv("data_sets/validation_set.csv"))
    df_test  = clean_column_names(pd.read_csv("data_sets/test_set.csv"))
    
    target_column = "Match_Ready"

    # -----------------------
    # 2. Preprocessing
    # -----------------------
    preprocessing_rules = fit_preprocessing(df_train, target_column)

    X_train, y_train = preprocessing_data(df_train, preprocessing_rules, target_column)
    X_val, y_val     = preprocessing_data(df_val, preprocessing_rules, target_column)
    X_test, y_test   = preprocessing_data(df_test, preprocessing_rules, target_column)

    # -----------------------
    # 3. Train model
    # -----------------------
    model = MyLogisticRegression(learning_rate=0.01, iterations=5000, monitor_training=False)
    model.fit(X_train.values, y_train.values)
    print("✅ Model trained.\n")

    # -----------------------
    # 4. Test predictions
    # -----------------------
    datasets = [("Training", X_train, y_train),
                ("Validation", X_val, y_val),
                ("Test", X_test, y_test)]

    for name, X, y in datasets:
        preds = model.predict(X.values)
        accuracy = (preds == y.values).mean()
        print(f"{name} set accuracy: {accuracy:.2%}")
        print(f"Predictions: {preds}")
        print(f"True labels: {y.values}\n")

    # -----------------------
    # 5. Spot-check non-match ready player
    # -----------------------
    non_ready_idx = y_train[y_train == 0].index[0]
    sample = X_train.iloc[non_ready_idx:non_ready_idx+1]
    pred = model.predict(sample.values)[0]
    print(f"Spot check training non-ready sample prediction: {pred} (true: 0)")

if __name__ == "__main__":
    test_model()
