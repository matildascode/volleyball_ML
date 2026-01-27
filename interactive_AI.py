import pandas as pd
import numpy as np
from preprocessing import clean_column_names, fit_preprocessing, preprocessing_data
from train_model import MyLogisticRegression

FEATURE_ADVICE = {
    "Sleep_Hours": "Increase sleep duration and aim for consistent sleep quality.",
    "Resting_HR": "Elevated resting heart rate suggests incomplete recovery. Prioritise rest.",
    "HRV": "Low HRV indicates nervous system fatigue. Consider lighter training and recovery work.",
    "Fatigue_Level": "High fatigue suggests overload. Reduce training intensity and volume.",
    "Muscle_Soreness": "High soreness indicates muscle damage. Focus on recovery and mobility.",
    "Stress_Level": "High stress negatively impacts readiness. Use relaxation or recovery techniques.",
    "Last_Training_Duration_Mins": "Long recent sessions may impair readiness. Shorten or reduce intensity.",
    "Last_Session_RPE": "High perceived exertion suggests overload. Schedule an easier session.",
    "Hydration_Level": "Low hydration reduces performance. Increase fluid intake.",
    "Agility_Test_Time": "Slower agility times suggest neuromuscular fatigue. Emphasise recovery."
}

def model_advice(player_row, model, preprocessing_rules, threshold=0.5):
    player_df = pd.DataFrame([player_row])
    X_player, _ = preprocessing_data(player_df, preprocessing_rules)

    pred,confidence = model.predict(X_player.values)

    if pred == 1:
        print(f"✅ You are match ready! (confidence: {confidence[0]:.2f})")
        return

    print(f"❌ You are NOT match ready (confidence: {confidence[0]:.2f})")

    # Contribution to decision
    contributions = X_player.values.flatten() * model.weights

    feature_effects = list(zip(X_player.columns, contributions))

    # Sort by most negative impact
    feature_effects.sort(key=lambda x: x[1])

    print("\n🔍 Main factors reducing readiness:")

    for feature, impact in feature_effects[:5]:
        if impact < 0:
            print(f"- {feature} is negatively affecting readiness")

    print("\n💡 Suggested focus areas:")
    shown = set()

    for feature, impact in feature_effects:
        base_feature = feature.split("_")[0]

        if impact < 0 and base_feature in FEATURE_ADVICE and base_feature not in shown:
            print(f"- {FEATURE_ADVICE[base_feature]}")
            shown.add(base_feature)

    if not shown:
        print("- No single critical issue detected. Small improvements across recovery factors may help.")


