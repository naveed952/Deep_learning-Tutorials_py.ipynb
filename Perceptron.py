import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return np.where(
            self.sigmoid(self.weighted_sum(X)) >= 0.5,
            1,
            0
        )

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):

            error = 0

            for xi, target in zip(X, y):

                y_pred = self.predict(xi)

                update = self.eta * (target - y_pred)

                self.w_[1:] = self.w_[1:] + update * xi
                self.w_[0] = self.w_[0] + update

                error += int(update != 0.0)

            self.errors_.append(error)


# --------------------------------------------------
# Load Iris dataset
# --------------------------------------------------

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None
)

df = shuffle(df)

print("\nFirst 5 rows of Iris dataset:")
print(df.head())


# --------------------------------------------------
# Separate features and labels
# --------------------------------------------------

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst X row:")
print(X[0])

print("\nFirst y value:")
print(y[0])


# --------------------------------------------------
# Split data into training and testing data
# --------------------------------------------------

train_data, test_data, train_labels, test_labels = train_test_split(
    X,
    y,
    test_size=0.25
)

print("\nTraining data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

print("\nTraining labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)


# --------------------------------------------------
# Convert labels
# Iris-setosa = 1
# Everything else = 0
# --------------------------------------------------

train_labels = np.where(
    train_labels == 'Iris-setosa',
    1,
    0
)

test_labels = np.where(
    test_labels == 'Iris-setosa',
    1,
    0
)

print("\nConverted training labels:")
print(train_labels)

print("\nConverted testing labels:")
print(test_labels)


# --------------------------------------------------
# Create and train Perceptron
# --------------------------------------------------

model = Perceptron(
    eta=0.1,
    n_iter=10
)

model.fit(train_data, train_labels)

print("\nTraining completed!")

print("Learned weights:")
print(model.w_)

print("Errors per iteration:")
print(model.errors_)


# --------------------------------------------------
# Predict test data
# --------------------------------------------------

test_predictions = model.predict(test_data)

print("\nTest predictions:")
print(test_predictions)

print("\nActual test labels:")
print(test_labels)


# --------------------------------------------------
# Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

print("\nAccuracy:", round(accuracy, 2) * 100, "%")


# --------------------------------------------------
# Manual prediction
# --------------------------------------------------

print("\nEnter flower measurements:")

sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

sample = np.array([
    [sepal_length, sepal_width, petal_length, petal_width]
])

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction: Iris-setosa")
else:
    print("Prediction: Not Iris-setosa")