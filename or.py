import numpy as np

class PerceptronOR:
    def __init__(self):
        self.weights = np.array([0.5, 0.5])  # Pesos iniciales
        self.bias = -0.2                     # Sesgo

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)

# Datos de entrenamiento OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# Entrenamiento (regla de aprendizaje)
perceptron_or = PerceptronOR()
for epoch in range(10):
    for x, y_true in zip(X_or, y_or):
        y_pred = perceptron_or.predict(x)
        error = y_true - y_pred
        perceptron_or.weights += error * x
        perceptron_or.bias += error

# Prueba
print("Predicciones OR:", [perceptron_or.predict(x) for x in X_or])
# Salida esperada: [0, 1, 1, 1]
