import pandas as pd
import numpy as np
from preprocessing import clean_column_names, fit_preprocessing, preprocessing_data
from train_model import MyLogisticRegression

def model_advice(player_row, model, preprocessing_rules, threshold=0.5):
    player_df = pd.DataFrame([player_row])
    X_player, _ = preprocessing_data(player_df, preprocessing_rules)

    pred = model.predict(X_player.values, threshold=threshold)[0]

    if pred == 1:
        print("✅ You are match ready!")
        return

    z_current = np.dot(X_player.values, model.weights) + model.bias
    z_target = np.log(threshold / (1 - threshold))
    # Difference needed
    delta_z = z_target - z_current

    feature_impact = model.weights * X_player.values.flatten()
    advice = {}

    for i, in enumerate(X_player.columns):
        w = model.weights[i]
        if w !=0:
            delta_x = delta_z / w
            if w > 0:
                advice[feature] = f"Increase {feature} by {delta_x} to become match ready :)"
            else:
                advice[feature] = f"Decrease {feature} by {delta_x} to become match ready :)"

    print("❌ You are not match ready.")
    print("Feature adjustments to become ready:")
    for msg in advice.values():
        print("-", msg)

    
