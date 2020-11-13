import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt


df = pd.read_csv("../diabetes.csv")


def add_plot(outcome: int, mark: str, label: str, color: str = "black"):
    kde_kws_attrs = {"linestyle": mark, "color": color, "label": label}
    sns.distplot(df.loc[df.Outcome == outcome][col], hist=False, axlabel=False, kde_kws=kde_kws_attrs)

plt.subplots(3, 3, figsize=(15, 15))
for idx, col in enumerate(df.columns):
    ax = plt.subplot(3, 3, idx+1)
    ax.yaxis.set_ticklabels([])
    add_plot(0, "-", "No Diabetes", "red")
    add_plot(1, "--", "Diabetes", "black")
    ax.set_title(col)

plt.subplot(3, 3, 9).set_visible(False)
plt.show()
