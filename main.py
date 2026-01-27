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

    model = MyLogisticRegression()
    model.fit(X_train.values, y_train.values)
    model.accuracy_test(X_train, y_train,X_val, y_val, X_test, y_test)
    print("\n✅ Training complete!\n")

    valid_menstral_phases = {"Menstrual_Phase": ["FOLLICULAR", "OVULATORY", "LUTEAL", "MENSTRUAL"]}

    print("Enter player information to predict match readiness:")
    player_input = {}
    for col in df_train.drop(columns=[target_column]).columns:
        value = None
        while value is None:
            user_input = input(f"{col}: ")

            if col in preprocessing_rules["numeric_cols"]:
                try:
                    value = float(user_input)
                except ValueError:
                    print(f"❌ Invalid input. Please enter a number for {col}.")
                    value = None 
                if value < 0:
                    print(f"❌ Invalid input. Please enter a non negative number for {col}.")
                    value = None
            else:
                if col == "Menstrual_Phase":
                    if user_input.upper() not in valid_menstral_phases[col]:
                        print(f"❌ Invalid input. Please choose one of:")
                        print(' '.join([item.capitalize() for item in valid_menstral_phases[col]]))
                        value = None
                    else:
                        value = user_input.capitalize()
                else:
                    # Accept any string for other categorical features
                    value = user_input
        player_input[col] = value

    model_advice(player_input, model, preprocessing_rules)

if __name__ == "__main__":
    main()
