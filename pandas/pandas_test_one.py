import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# url: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
df = pd.read_csv("iris.data", names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"]
)

marker_shapes = [".", "^", "*"]

# Make scatter plot
ax = plt.axes()
for i, species in enumerate(df["class"].unique()):
    species_data = df[df["class"] == species]
    species_data.plot.scatter(x="sepal_length",
                              y="sepal_width",
                              marker=marker_shapes[i],
                              s=100,
                              title="Sepal Width vs Length by Species",
                              label=species, figsize=(10, 7), ax=ax
                              )

random_index = np.random.choice(df.index, replace=False, size=10)
df.loc[random_index,'sepal_length'] = None
print(df.isnull().any())
print(f"Before {df.shape[0]}")
df2 = df.dropna()
print(f"After {df2.shape[0]}")
df.sepal_length = df.sepal_length.fillna(df.sepal_length.mean())
print(f"Far After {df.shape[0]}")

plt.show()
