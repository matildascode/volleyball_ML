import pandas as pd
import numpy as np

class MyLogisticRegression :
    """Binary logistic regression trained with batch gradient descent.

    Implemented from scratch with NumPy (no scikit-learn) to predict whether
    a player is match-ready. Minimises binary cross-entropy loss.

    Attributes:
        learning_rate: Step size for each gradient descent update.
        iterations: Number of gradient descent steps during training.
        monitor_training: Whether to print the loss while training.
        weights: Learned feature coefficients, shape (n_features,).
            None until ``fit`` is called.
        bias: Learned intercept term. None until ``fit`` is called.
    """

    def __init__(self, learning_rate=0.1, iterations=10000, monitor_training = True):
        """Sets hyperparameters and initliazes model parameters

        Args: learnings_rate: step sixe for gradient descent
              iterations: number of optimisation steps
              monitor_training: If True print the loss every 1000 iterations

        """

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.monitor_training = monitor_training
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Converts logg-odds into a probability

        Args: z: linear model outpuy (scalar or array)
        Returns: probabilities with the same shape as 'z'
        """

        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the model with gradient descent on binary cross-entropy loss.

        Args: X: feature matrix (n_samples x n_features)
              y: Binary vector of 0s and 1s, shape (n_samples,), where
                 y[i] is 1 if sample i is match-ready
        Returns: None
        """

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.iterations):

            # Linear combinations of features: z = X.w + b
            linear_output = np.dot(X, self.weights) + self.bias

            y_pred = self.sigmoid(linear_output) 
            # Avoid log(0) by making sure y_pred is in correct range
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)


            # Binary cross-entropy loss: penalises confident predictions that disagree with y
            loss = -np.mean( y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

            if self.monitor_training and i % 1000 == 0:
                print(f"Iteration {i} has a loss of {loss}")

            # Gradient decent
            # How much each feature(dw) and bias(db) should be increased/decreased
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Step downhill, scaled by learning_rate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X, threshold = 0.5):
        """Predicts class labels and probabilities for new data.
        
        Args: X: feature matric preprocessed
              threshold: probobility at or above which a sample is calssed as 1 (match ready), deault= 0.5
        Returns: tuple (predictions, probabilities): binary labels and the predicted probability of match
                 readyness for each sample

        """

        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        predictions = np.zeros_like(probabilities, dtype=int)

        # Assign 1 to any probability above the threshold (default is 0)
        predictions[probabilities >= threshold] = 1

        return predictions,probabilities
    
    def accuracy_test(self,X_train, y_train,X_val, y_val, X_test, y_test):
        """ Print accuracy for the traning, validation and test sets.

        Args: X_train, X_val, X_test: preprocessed feature dataframes.
              y_train, y_val, y_test: matching labels
        Returns: None.
        """

        datasets = [("Training", X_train, y_train),("Validation", X_val, y_val),("Test", X_test, y_test)]

        for name, X, y in datasets:
            preds = self.predict(X.values)[0]
            accuracy = (preds == y.values).mean()
            print(f"{name} set accuracy: {accuracy:.2%}")
            print(f"Predictions: {preds}")
            print(f"True labels: {y.values}\n")

