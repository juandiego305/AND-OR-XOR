import numpy as np
from keras.models import Sequential
from keras.layers import Dense

X_xor = np.array([0,0], [0,1],[1,0],[1,1])
Y_xor = np.array([0,1,1,0])

model = Sequential([
    Dense(2,activation ='reli', input_dim = 2),
    Dense(1,activation = 'sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_xor, Y_xor, epochs= 1000, verbose= 0)

predictions = np.round(model.predict(X_xor).flatten())
print("predicciones XOR:", predictions.astype(int))
