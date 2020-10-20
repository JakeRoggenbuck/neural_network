import pandas as pd
import matplotlib.pyplot as plt


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

# Make other plots
df.plot.hist("Hey")
df.plot.box("Hey")
plt.show()
