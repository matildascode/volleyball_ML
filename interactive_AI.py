"""Interactive readiness advice built from the trained model's weights.

For a player the model predicts is NOT match-ready, each actionable feature's
contribution to the decision is calculated, and advice is shown for the
features that pull the player's readiness down the most.
"""

import pandas as pd
import numpy as np
from apply_preprocessing import clean_column_names, fit_preprocessing, preprocessing_data
from train_model import MyLogisticRegression

# Advice only exists for features a player can act on.
# Age, height andmenstrual phase are deliberately absent: they can't be change

FEATURE_ADVICE = {
    "Sleep_Hours": "Increase sleep duration and aim for consistent sleep quality.",
    "Resting_HR": "Elevated resting heart rate suggests incomplete recovery. Prioritise rest.",
    "Fatigue_Level": "High fatigue suggests overload. Reduce training intensity and volume.",
    "Muscle_Soreness": "High soreness indicates muscle damage. Focus on recovery and mobility.",
    "Stress_Level": "High stress negatively impacts readiness. Use relaxation or recovery techniques.",
    "Last_Training_Duration_Mins": "Long recent sessions may impair readiness. Shorten or reduce intensity.",
    "Pre_Match_Injury_Flag": "An injury flag lowers readiness. Get it assessed by a physio or coach before playing.",
}

MAX_FACTORS_SHOWN = 5

def get_negative_factors(X_player, weights):
    """Rank the actionable features that reduce a player's readiness.

    Features are standardised, so 'value * weight' shows how far a feature
    pushes the prediction away from an average player. Negative values push
    towards "not match-ready".

    Args: X_player: preprocessed single-row feature DataFrame
          weights: learned model weights, one per column of 'X_player'

    Returns: List of (feature_name, contribution) tuples with negative contributions only, most harmful first
    """

    contributions = X_player.values.flatten() * weights

    return sorted(
        ((feature, contribution)
            for feature, contribution in zip(X_player.columns, contributions)
            if feature in FEATURE_ADVICE and contribution < 0
        ),
        key=lambda item: item[1],)

def model_advice(player_row, model, preprocessing_rules, threshold=0.5):
    """Predict a player's match readiness and print personalised advice.

    Args: player_row: raw feature values keyed by column name (target excluded)
          model: fitted 'MyLogisticRegression'
          preprocessing_rules: rules returned by fit_preprocessing
          threshold: Probability at or above which a player counts as ready

    Returns:  None. Results are printed.
    """

    player_df = pd.DataFrame([player_row])
    X_player, _ = preprocessing_data(player_df, preprocessing_rules)

    predictions, probabilities = model.predict(X_player.values, threshold=threshold)
    prob_ready = probabilities[0]

    if predictions[0] == 1:
        print(f"✅ You are match ready! (probability of readiness: {prob_ready:.0%})")
        return

    print(f"❌ You are NOT match ready (probability of readiness: {prob_ready:.2f})")

    negative_factors = get_negative_factors(X_player, model.weights)

    if not negative_factors:
        print("\n💡 No single critical issue detected. Small improvements across recovery factors may help.")
        return

    print("\n🔍 Main factors reducing readiness, with suggested focus areas:")
    for feature, _ in negative_factors[:MAX_FACTORS_SHOWN]:
        print(f"- {feature.replace('_', ' ')}: {FEATURE_ADVICE[feature]}")


