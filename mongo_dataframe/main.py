from jakesutils.database import Database
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np

from sklearn.model_selection import train_test_split

import math


# Connect to database
db = Database()
db.connect()

client = db.client
database = client.server_test_data
raw_obj_pit = database.raw_obj_pit

# Make dataframe from database
raw_df = pd.DataFrame(list(raw_obj_pit.find()))
df = raw_df.loc[
    :,
    [
        "can_cross_trench",
        "drivetrain_motor_type",
        "drivetrain_motors",
        "has_ground_intake",
    ],
]

df["has_ground_intake"] = df["has_ground_intake"].astype(int)

# Make model
model = Sequential()
model.add(Dense(units=3, activation="sigmoid", input_dim=3))
model.add(Dense(units=1, activation="sigmoid"))

# Make optimizer
sgd = optimizers.SGD(lr=1)
model.compile(loss="mean_squared_error", optimizer=sgd)

# Split train and test data
train, test = train_test_split(df, test_size=0.2, shuffle=True)

X = train.loc[:, ["drivetrain_motor_type", "drivetrain_motors", "has_ground_intake"]]
y = train.loc[:, ["can_cross_trench"]]

model.fit(X, y, epochs=1500, verbose=False)
print(model.predict(X))

# Test the model
X_test = test.loc[:, ["drivetrain_motor_type", "drivetrain_motors", "has_ground_intake"]]
y_test = test.loc[:, ["can_cross_trench"]]

data_X_top = X_test.head()
data_y_top = y_test.head()

for n in range(len(data_X_top.values)):
    predict_data = data_X_top.values[n]

    actual = data_y_top.values[n][0]
    guess = model.predict(np.array([predict_data]))[0][0]

    print(f"{predict_data} -> {guess}: {actual}")
