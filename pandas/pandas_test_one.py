import pandas as pd


# url: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
df = pd.read_csv("iris.data", names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"]
)

print(df.info())
print(df.describe())
print(df.head(10))
