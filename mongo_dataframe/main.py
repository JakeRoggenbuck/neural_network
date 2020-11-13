from jakesutils.database import Database
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np

from sklearn.model_selection import train_test_split

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

# Make model
model = Sequential()
model.add(Dense(units=7, activation="sigmoid", input_dim=7))
model.add(Dense(units=1, activation="sigmoid"))

# Make optimizer
sgd = optimizers.SGD(lr=1)
model.compile(loss="mean_squared_error", optimizer=sgd)

# Split train and test data
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
