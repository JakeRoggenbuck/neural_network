from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(units=4000, activation='sigmoid', input_dim=3))
model.add(Dense(units=1000, activation='sigmoid'))
print(model.summary())
