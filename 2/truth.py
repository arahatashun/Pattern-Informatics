import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

class_lists = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
df = pd.read_csv("iris.csv")
data = df.values
numberOfData = data.shape[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hot = plt.get_cmap('hot')
cNorm = colors.Normalize(vmin=0, vmax=len(class_lists))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=hot)

for i in range(3):
    x = data[data[:, 4] == class_lists[i]][:, 0]
    y = data[data[:, 4] == class_lists[i]][:, 1]
    z = data[data[:, 4] == class_lists[i]][:, 2]
    k_label = class_lists[i]
    ax.scatter(x, y, z, s=15, color=scalarMap.to_rgba(i), label=k_label)

ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal length')
ax.grid(True)
ax.legend(loc='upper left')
ax.set_title('truth')

filename = "../picture/" + "truth" + ".png"
plt.savefig(filename)
plt.show()
