from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np


model = Sequential()
model.add(Dense(units=7, activation='sigmoid', input_dim=7))
model.add(Dense(units=1, activation='sigmoid'))

sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

np.random.seed(9)

X = np.array([[1,1,1,0,1,1,1], [1,1,1,0,1,1,1], [1,1,1,0,1,1,1], [1,1,1,0,1,1,1], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,0,0,0,0,0]])
y = np.array([[0],[0],[0],[0],[1],[1],[1]])
model.fit(X, y, epochs=1500, verbose=False)
print(model.predict(X))

j = np.array([[1,1,1,0,1,1,0]])
print(model.predict(j))
