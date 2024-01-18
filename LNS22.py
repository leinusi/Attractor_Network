import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import random


digits = load_digits()
X, y = digits.data, digits.target
X /= X.max()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class AttractorNetwork:
    def __init__(self, num_cells, num_classes):
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.weights = np.random.rand(num_cells, num_classes)
        self.adjacency_matrix = self.create_adjacency_matrix()
        self.best_loss = np.inf
        self.epochs_completed = 0

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

    def load_weights(self, file_path):
        self.weights = np.load(file_path)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def update_cell_states(self, x):
        return np.dot(self.adjacency_matrix, x) / 4

    def train(self, X_train, y_train, learning_rate, epochs, stop_loss=0.5):
      while self.epochs_completed < epochs:
        best_loss = np.inf
        epochs_without_improvement = 0
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

            self.epochs_completed += 1
            if avg_loss <= stop_loss:
                print(f"Training stopped because the loss reached the specified threshold: {stop_loss}")
                np.save("model_weights.npy", self.weights)
                return True

        return False


    def predict(self, X_test):
        predictions = []
        for x in X_test:
            x = self.update_cell_states(x)
            cell_output = np.dot(x, self.weights)
            cell_output_softmax = self.softmax(cell_output.reshape(1, -1))
            predicted_class = np.argmax(cell_output_softmax)
            predictions.append(predicted_class)
        return predictions

# 定义遗传算法中的辅助函数
def fitness_function(individual, nn, X_train, y_train, learning_rate, epochs):
    nn.train(X_train, y_train, learning_rate, epochs)
    y_pred = nn.predict(X_train)
    return accuracy_score(y_train, y_pred)

def genetic_algorithm(nn, X_train, y_train, learning_rate, epochs, population_size, num_generations, stop_loss=0.05):
    max_fitness_per_generation = []
    avg_fitness_per_generation = []

    num_features = nn.num_cells * nn.num_classes
    crossover_rate = 0.7
    mutation_rate = 0.1

    population = [np.random.choice([0, 1], size=(num_features,)) for _ in range(population_size)]

    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations} in progress...")

        fitness_scores = [fitness_function(individual, nn, X_train, y_train, learning_rate, epochs) for individual in population]

        max_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        max_fitness_per_generation.append(max_fitness)
        avg_fitness_per_generation.append(avg_fitness)

        selected_individuals = random.choices(population, weights=fitness_scores, k=population_size)

        next_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_individuals[i], selected_individuals[i+1]
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, num_features - 1)
                offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            else:
                offspring1, offspring2 = parent1, parent2
            next_population.extend([offspring1, offspring2])

        for individual in next_population:
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, num_features - 1)
                individual[mutation_point] = 1 - individual[mutation_point]

        population = next_population

        if nn.train(X_train, y_train, learning_rate, epochs, stop_loss):
            print(f"Training stopped early at generation {generation + 1} due to loss threshold.")
            break

    print("Max Fitness per Generation:", max_fitness_per_generation)
    print("Average Fitness per Generation:", avg_fitness_per_generation)

    plt.plot(max_fitness_per_generation, label='Max Fitness')
    plt.plot(avg_fitness_per_generation, label='Average Fitness')
    plt.legend()
    plt.show()

    best_individual = max(population, key=lambda ind: fitness_function(ind, nn, X_train, y_train, learning_rate, epochs))
    return best_individual

if __name__ == "__main__":
    num_cells = 64
    num_classes = 10
    learning_rate = 0.01
    epochs = 100

    nn = NeuralNetwork(num_cells, num_classes)
    best_rule = genetic_algorithm(nn, X_train, y_train, learning_rate, epochs, population_size=20, num_generations=50, stop_loss=0.05)

    nn.load_weights("model_weights.npy")

    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
