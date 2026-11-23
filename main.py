"""Entry point for the Volleyball Match-Readiness model.

Pipeline:
    1. Load the training, validation and test sets.
    2. Fit preprocessing on the training set and apply it to all three sets.
    3. Train the from-scratch logistic regression model.
    4. Report accuracy, then take a player's details and give advice.
"""

import pandas as pd
from apply_preprocessing import clean_column_names, fit_preprocessing, preprocessing_data
from train_model import MyLogisticRegression
from interactive_AI import model_advice

DATA_DIR = "data_sets"
TRAIN_FILE = f"{DATA_DIR}/training_set.csv"
VAL_FILE = f"{DATA_DIR}/validation_set.csv"
TEST_FILE = f"{DATA_DIR}/test_set.csv"
TARGET_COLUMN = "Match_Ready"

VALID_MENSTRUAL_PHASES = ["Follicular", "Ovulatory", "Luteal", "Menstrual"]

# Min and max for each numeric input. Rejects typos 
#Columns not listed here only need to be non-negative.
NUMERIC_LIMITS = {
    "Age": (16, 40),
    "Height_cm": (140, 220),
    "Sleep_Hours": (0, 24),
    "Resting_HR": (30, 120),
    "Fatigue_Level": (1, 10),
    "Muscle_Soreness": (1, 10),
    "Stress_Level": (1, 10),
    "Last_Training_Duration_Mins": (0, 300),
    "Pre_Match_Injury_Flag": (0, 1),
}

def load_data(path: str):
    """Read a CSV file and tidy its column names."""
    return clean_column_names(pd.read_csv(path))

def read_number(column: str):
    """Prompt until the user enters a number within the allowed range."""
    low, high = NUMERIC_LIMITS.get(column, (0, float("inf")))
    while True:
        raw = input(f"{column}: ")
        try:
            value = float(raw)
        except ValueError:
            print(f"❌ Invalid input. Please enter a number for {column}.")
            continue

        if not low <= value <= high:
            print(f"❌ Invalid input. {column} must be between {low} and {high}.")
            continue
        return value

def read_menstrual_phase():
    """Prompt until the user enters a valid menstrual phase (case-insensitive)."""
    while True:
        raw = input("Menstrual_Phase: ").strip().capitalize()
        if raw in VALID_MENSTRUAL_PHASES:
            return raw
        print("❌ Invalid input. Please choose one of:")
        print(" ".join(VALID_MENSTRUAL_PHASES))


def read_player_input(feature_columns: list, numeric_cols: list) -> dict:
    """Ask the user for one value per model feature.

    Args:feature_columns: feature names, in the order they are asked
         numeric_cols: names of numeric features (from the preprocessing rules)
    Returns: dictionary mapping each feature name to the validated value
    """
    player_input = {}
    for col in feature_columns:
        if col in numeric_cols:
            player_input[col] = read_number(col)
        elif col == "Menstrual_Phase":
            player_input[col] = read_menstrual_phase()
        else:
            player_input[col] = input(f"{col}: ")
    return player_input


def main():
    # Load data 
    df_train = load_data(TRAIN_FILE)
    df_val = load_data(VAL_FILE)
    df_test = load_data(TEST_FILE)

    # Preprocess: learn rules from training data only, apply to all sets
    preprocessing_rules = fit_preprocessing(df_train, TARGET_COLUMN)
    X_train, y_train = preprocessing_data(df_train, preprocessing_rules, TARGET_COLUMN)
    X_val, y_val = preprocessing_data(df_val, preprocessing_rules, TARGET_COLUMN)
    X_test, y_test = preprocessing_data(df_test, preprocessing_rules, TARGET_COLUMN)

    # Train and evaluate
    model = MyLogisticRegression()
    model.fit(X_train.values, y_train.values)
    model.accuracy_test(X_train, y_train, X_val, y_val, X_test, y_test)
    print("\n✅ Training complete!\n")

    # Interactive advice
    print("Enter player information to predict match readiness:")
    feature_columns = df_train.drop(columns=[TARGET_COLUMN]).columns
    player_input = read_player_input(feature_columns, preprocessing_rules["numeric_cols"])
    model_advice(player_input, model, preprocessing_rules)


if __name__ == "__main__":
    main()