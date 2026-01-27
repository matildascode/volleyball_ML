Volleyball Match-Readiness Prediction:
An end-to-end Machine Learning pipeline using custom Logistic Regression to predict athlete availability and provide actionable recovery advice.

Project Overview
In high-performance volleyball, "availability" is a competitive advantage. This project uses a custom-built Logistic Regression model to predict whether an athlete is match-ready based on a combination of biometric, training, and wellness data.

Unlike a "black-box" model, this system provides interpretability, allowing coaches to see exactly which factors (e.g Sleep, Stress, or HRV) are negatively impacting a player's readiness.

The Data
The model is trained and validated on three distinct datasets: training_set.csv, validation_set.csv, and test_set.csv

    Features Included:
    Biometrics: Age, Height (cm), Resting HR, HRV (Heart Rate Variability).
    Load/Performance: Last Training Duration (mins), Last Session RPE (Perceived Exertion).
    Wellness/Recovery: Sleep Hours, Fatigue Level, Muscle Soreness, Stress Level.
    Domain Specific: Menstrual Phase, Pre-Match Injury Flag.
    Target: Match_Ready (Binary: 1 for Ready, 0 for Not Ready).

Technical Architecture
1. Custom Logistic Regression (train_model.py)

I implemented a Logistic Regression class from scratch using NumPy to demonstrate a deep understanding of the underlying mathematics.
 . Loss Function: Binary Cross-Entropy.
 . Optimization: Gradient Descent with adjustable learning rate and iterations.
 . Sigmoid Activation: To map linear combinations to probabilities.

You can also monitor training by setting monitor_training = True (default) as a parameter for MyLogisticRegression() class.
It outputs the 1000'th iterations binary cross entropy loss, when run, at the start the loss is 0.6931(3dp), but by the 9000's iteration of training it has a loss of 0.0016(3dp)

As part of the MyLogisticRegression() class you can run the function accuracy_test where the model makes predictions on each row of the 3 data sets.
It calaculate what percentage it was correct and which show you the models redictions compared to the true label for each row of data ( shown in main file)


2. Robust Preprocessing (preprocessing.py)

To ensure data quality, the pipeline includes:

    Median Imputation: Automatically handles missing numeric data.
    One-Hot Encoding: Converts categorical features like Menstrual_Phase into numerical inputs.
    Standard Scaling: Normalizes features so that variables with different scales (like Height vs. HRV) are weighted fairly by the model.

3. Interactive AI Advice (interactive_AI.py)

This project goes beyond prediction by offering a coaching recommendation system. When a player enters their information and is flagged as "Not Ready," the system calculates the feature contributions to identify the top 5 negative impacts and suggests specific recovery actions.

Getting Started
Prerequisites
 - Python 3.x
 - Pandas, NumPy, Scikit-learn

Installation
git clone https://github.com/your-username/volleyball-readiness.git
cd volleyball-readiness
pip install -r requirements.txt

Usage
Run the main script to train the model and enter player data for a real-time prediction:
    python main.py
