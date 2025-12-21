from preprocessing import clean_column_names, fit_preprocessing, preprocessing_data
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_train = clean_column_names(pd.read_csv("training_set.csv"))
df_val = clean_column_names(pd.read_csv("validation_set.csv"))

preprocessing_rules = fit_preprocessing(df_train, "Match_Ready")

X_val, y_val = preprocessing_data(df_val, preprocessing_rules, "Match_Ready")

class MyLogisticRegression :

    def __init__(self, learning_rate=0.01, iterations=1000, monitor_training = True):
        '''
        Sets hyperparameters and initliazes model parameters

        Hyperparameters:
        
        learning rate (float): controls the size of the gradient
        interating (int): number of optimization steps to train the model

        Model Parameters:

        monitor_training (bool): if True, prints the loss during training to monitor progress
        bias (float): default tendency alowing the model to shift the decisio boundry

        '''

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.monitor_training = monitor_training
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        '''
        Converts logg-odds into a probability
        '''
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        '''
        Train the model using binary cross-entropy loss and gradient descent
        
        X: feature matrix (n_samples x n_features)
        y: target vector (n_samples)
        '''
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.iterations):

            # Linear combinations of features
            linear_output = np.dot(X, self.weights) + self.bias

            y_pred = self._sigmoid(linear_output) 

            # Binary cross-entropy loss
            loss = -np.mean( y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

            if self.monitor_training:
                print(f"Iteration {i} has a loss of {loss}")

            # Gradient decent
            # How much each feature(dw) and bias(db) should be increased/decreased
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def prediction(self, X):
        '''
        Makes predictions on new data in binary output
        '''
        predictions = np.zeros_like(probabilities, dtype=int)

        z = np.dot(X, self.weights) + self.bias
        probability = self._sigmoid(z)

        # Assign 1 to any probability above the threshold (default is 0)
        predictions[probabilities >= threshold] = 1

        return predictions


