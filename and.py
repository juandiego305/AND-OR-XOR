from sklearn.linear_model import Perceptron
# Datos AND
X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [0, 0, 0, 1]

# Entrenamiento
perceptron_and = Perceptron (max_iter=100, random_state=42) 
perceptron_and.fit(X_and, y_and)
# Predicciones
print("Predicciones AND: ", perceptron_and.predict(X_and)) # Salida esperada: [0, 0, 0, 1]
