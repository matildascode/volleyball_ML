Volleyball Match-Readiness Prediction:
An end-to-end machine learning pipeline that uses a logistic regression model, built from scratch in NumPy, to predict whether a volleyball player is match-ready, and gives personalised recovery advice when she is not.

Project Overview
In high-performance volleyball, availability is a competitive advantage. This project predicts match readiness from training, recovery and wellness data.

Unlike a black-box model, the weights of a logistic regression can be read directly. The advice system uses them to show which factors (for example sleep, stress or muscle soreness) are pulling a player's readiness down.

Results



The Data
The model uses three datasets in data_sets/: training_set.csv, validation_set.csv and test_set.csv.

The data is synthetic. It was generated from rules I defined: wellness and load features move together, and Match_Ready is then drawn from a noisy probability, so a rested player is usually, but not always, ready. Noise was added deliberately so the task is not trivially separable. No real athlete data was used.

    Features Included:
    Biometrics: Age, Height (cm), Resting HR.
    Load/Performance: Last Training Duration (mins).
    Wellness/Recovery: Sleep Hours, Fatigue Level, Muscle Soreness, Stress Level.
    Domain Specific: Menstrual Phase, Pre-Match Injury Flag.
    Target: Match_Ready (Binary: 1 for Ready, 0 for Not Ready).

Project Structure
volleyball-readiness/
├── data_sets/
│   ├── training_set.csv
│   ├── validation_set.csv
│   └── test_set.csv
├── apply_preprocessing.py   # imputation, one-hot encoding, scaling
├── train_model.py           # logistic regression from scratch
├── interactive_AI.py        # prediction and personalised advice
├── main.py                  # entry point
└── requirements.txt

Technical Architecture
1. Logistic regression from scratch (train_model.py)
Loss function: binary cross-entropy, with predictions clipped to avoid log(0).
Optimisation: batch gradient descent with adjustable learning_rate and iterations.
Activation: sigmoid, to turn the linear output into a probability.
Monitoring: with monitor_training=True (the default), the loss is printed every 1000 iterations.
Evaluation: accuracy_test reports accuracy on the training, validation and test sets.

2. Preprocessing (apply_preprocessing.py)
Preprocessing is split into a fit step and an apply step. Rules are learnt from the training set only and then applied to every dataset, which prevents data leakage.

Median imputation: fills missing numeric values with the training median.
One-hot encoding: converts Menstrual_Phase into numeric columns (Follicular is the baseline category).
Standard scaling: puts features on the same scale (for example Height in cm and Sleep Hours) so gradient descent converges and the weights are comparable.

3. Interactive advice (interactive_AI.py)
When a player enters her details and is predicted not match-ready, the system:

Multiplies each standardised feature by its learned weight to get its contribution to the decision.
Ranks the actionable features (sleep, resting HR, fatigue, soreness, stress, training duration, injury flag) by how much they reduce readiness.
Shows up to five of them with a suggested recovery action.

Age, height and menstrual phase are left out of the advice because they are not something a player can act on.

Inputs are validated: non-numeric or out-of-range values are rejected and re-prompted.

Getting Started
Prerequisites
 - Python 3.x
 - Pandas, NumPy, scikit-learn(scikit-learn is used only for StandardScaler)

Installation
git clone https://github.com/matildascode/volleyball_ML.git
cd volleyball_ML
pip install -r requirements.txt

Usage:
Run the main script to train the model and enter player data for a real-time prediction:
    python main.py

Limitations
The data is synthetic and small, so results should not be read as real-world performance.
Logistic regression is a linear model and cannot capture interactions between features.
The advice comes from learned weights and is not medical or coaching guidance.
Menstrual phase weights are relative to the Follicular baseline and reflect only the synthetic data.

Future Improvements
Train and validate on real, consented athlete data.
Tune learning_rate and iterations using the validation set.
Pplot the loss curve.