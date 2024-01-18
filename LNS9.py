import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

digits = load_digits()
X, y = digits.data, digits.target

X /= X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class AttractorNetwork:
    def __init__(self, num_cells, num_classes):
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.weights = np.random.rand(num_cells, num_classes)
        self.adjacency_matrix = self.create_adjacency_matrix()

    def create_adjacency_matrix(self):
        matrix = np.zeros((self.num_cells, self.num_cells))
        for i in range(int(np.sqrt(self.num_cells))):
            for j in range(int(np.sqrt(self.num_cells))):
                index = i * int(np.sqrt(self.num_cells)) + j
                if i > 0:
                    matrix[index, index - int(np.sqrt(self.num_cells))] = 1
                if i < np.sqrt(self.num_cells) - 1:
                    matrix[index, index + int(np.sqrt(self.num_cells))] = 1
                if j > 0:
                    matrix[index, index - 1] = 1
                if j < np.sqrt(self.num_cells) - 1:
                    matrix[index, index + 1] = 1
        return matrix

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def update_cell_states(self, x):
        return np.dot(self.adjacency_matrix, x) / 4

    def train(self, X_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X_train)):
                x = self.update_cell_states(X_train[i])
                label = y_train[i]

                label_one_hot = np.zeros(self.num_classes)
                label_one_hot[label] = 1

                cell_output = np.dot(x, self.weights)
                cell_output_softmax = self.softmax(cell_output.reshape(1, -1))

                loss = -np.sum(label_one_hot * np.log(cell_output_softmax + 1e-7))
                total_loss += loss

                delta_weights = learning_rate * np.outer(x, (cell_output_softmax - label_one_hot).flatten())
                self.weights -= delta_weights

            avg_loss = total_loss / len(X_train)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            x = self.update_cell_states(x)
            cell_output = np.dot(x, self.weights)
            cell_output_softmax = self.softmax(cell_output.reshape(1, -1))
            predicted_class = np.argmax(cell_output_softmax)
            predictions.append(predicted_class)
        return predictions

num_cells = 64
num_classes = 10
learning_rate = 0.01
epochs = 100

an = AttractorNetwork(num_cells, num_classes)

an.train(X_train, y_train, learning_rate, epochs)

y_pred = an.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
