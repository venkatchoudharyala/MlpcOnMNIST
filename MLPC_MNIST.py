import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
y_train=TrainSplit.label

# considerations for x_train
TrainSplitX=TrainSplit.iloc[:, 1:785]
x_train=np.array(TrainSplitX)
y_test=TestSplit.label

# considerations for x_train
TestSplitX=TestSplit.iloc[:, 1:785]
x_test=np.array(TestSplitX)

# Reshape images to 28x28 pixels
x_train = x_train.reshape((x_train.shape[0], 28*28))
x_test = x_test.reshape((x_test.shape[0], 28*28))

# Create MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                      solver='sgd', verbose=10, random_state=1,
                      learning_rate_init=0.00001)

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels of the test data
y_pred = model.predict(x_test)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


