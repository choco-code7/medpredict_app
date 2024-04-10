import numpy as np
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, C=1.0, learning_rate=0.01, num_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.losses = []  # Initialize the list to store loss values

    def _objective(self, margins):
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - margins).sum()

    def fit(self, X, Y):
        N, D = X.shape
        self.w = np.random.randn(D)
        self.b = 0

        for _ in range(self.num_iterations):  # Use self.num_iterations
            margins = Y * self._decision_function(X)
            loss = self._objective(margins)
            self.losses.append(loss)  # Store loss

            idx = np.where(margins < 1)[0]
            grad_w = self.w - self.C * Y[idx].dot(X[idx])
            grad_b = -self.C * Y[idx].sum()

            self.w -= self.learning_rate * grad_w  # Use self.learning_rate
            self.b -= self.learning_rate * grad_b  # Use self.learning_rate

        self.support_ = np.where((Y * self._decision_function(X)) <= 1)[0]

    def _decision_function(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))
    
    def predict_with_confidence(self, X):
        distances = self._decision_function(X)
        signed_distances = distances * self.predict(X)  # Convert distances to signed distances

        # Calculate confidence as the inverse of distance from the decision boundary
        
        confidence = 1 / (1 + np.exp(-signed_distances))  # Apply sigmoid function for normalization

        # Convert confidence values from probabilities to percentages
        confidence_percentage = confidence * 100

        return self.predict(X), confidence_percentage
    


# Evaluation
    
    def accuracy(self, X, Y):
        P = self.predict(X)
        accuracy = np.mean(Y == P)
        return accuracy * 100 

    def precision(self, X, Y):
        P = self.predict(X)
        true_positives = np.sum((Y == 1) & (P == 1))
        false_positives = np.sum((Y == -1) & (P == 1))
        return true_positives / (true_positives + false_positives) * 100

    def recall(self, X, Y):
        P = self.predict(X)
        true_positives = np.sum((Y == 1) & (P == 1))
        false_negatives = np.sum((Y == 1) & (P == -1))
        return true_positives / (true_positives + false_negatives) * 100

    def f1_score(self, X, Y):
        prec = self.precision(X, Y)
        rec = self.recall(X, Y)
        return 2 * (prec * rec) / (prec + rec)

    def confusion_matrix(self, X, Y):
        P = self.predict(X)
        cm = np.zeros((2, 2), dtype=int)
        for true, pred in zip(Y, P):
            cm[int(true == 1), int(pred == 1)] += 1
        return cm
    def plot_confusion_matrix(self, X, Y):
        cm = self.confusion_matrix(X, Y)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = np.unique(Y)
        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted')
        plt.ylabel('True')

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.show()

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Loss per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
