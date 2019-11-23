import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors

df = pd.read_csv("auto_mpg.csv")
df_data = df.values

answerlabel_t = df_data.T[:][0]
answerlabel_t = np.matrix(answerlabel_t).T
pd.plotting.scatter_matrix(df, figsize=(7, 7))

# weight and horsepower plot
df_large_x = df
drop_col = ["mpg", "displacement", "acceleration"]
large_x = df_large_x.drop(drop_col, axis=1).values
x_zero = np.ones((large_x.shape[0], 1))
large_x = np.concatenate([x_zero, large_x], axis=1)
large_x = np.matrix(large_x)
inverse_matrix = np.linalg.inv(np.dot(large_x.T, large_x))
omega = np.dot(np.dot(inverse_matrix, (large_x.T)), answerlabel_t)
omega = np.array(omega)
# plot figure
fig = plt.figure(figsize=(7, 7))
ar_mpg = df_data.T[:][0]
ar_horsepower = df_data.T[:][2]
ar_weight = df_data.T[:][3]
ax = fig.add_subplot(111, projection='3d')
# set color
hot = plt.get_cmap('hot')
cNorm = colors.Normalize(vmin=answerlabel_t.min(), vmax=answerlabel_t.max())
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=hot)
# scatter plot
ax.scatter(ar_weight, ar_horsepower, ar_mpg, s=10, c=scalarMap.to_rgba(answerlabel_t)[:, 0, :])
# 3d plot
ax.set_xlabel('weight')
ax.set_ylabel('horsepower')
ax.set_zlabel('mpg')
ax.grid(True)
ax.legend(loc='upper left')
range_weight = np.arange(1500, 5000, 400)
range_horsepower = np.arange(50, 225, 30)
mesh_weight, mesh_horsepower = np.meshgrid(range_weight, range_horsepower)
mesh_mpg = omega[1] * mesh_horsepower + omega[2] * mesh_weight + omega[0]
ax.plot_wireframe(mesh_weight, mesh_horsepower, mesh_mpg)
plt.title("linear regression")
plt.show()

# displacement and horsepower linear regression
df_large_x = df
drop_col = ["mpg", "acceleration", "weight"]
large_x = df_large_x.drop(drop_col, axis=1).values
x_zero = np.ones((large_x.shape[0], 1))
large_x = np.concatenate([x_zero, large_x], axis=1)
large_x = np.matrix(large_x)
inverse_matrix = np.linalg.inv(np.dot(large_x.T, large_x))
omega = np.dot(np.dot(inverse_matrix, (large_x.T)), answerlabel_t)
omega = np.array(omega)
# plot figure
fig = plt.figure(figsize=(7, 7))
ar_mpg = df_data.T[:][0]
ar_horsepower = df_data.T[:][2]
ar_displacemnt = df_data.T[:][1]
ax = fig.add_subplot(111, projection='3d')
# set color
hot = plt.get_cmap('hot')
cNorm = colors.Normalize(vmin=answerlabel_t.min(), vmax=answerlabel_t.max())
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=hot)
# scatter plot
ax.scatter(ar_displacemnt, ar_horsepower, ar_mpg, s=10, c=scalarMap.to_rgba(answerlabel_t)[:, 0, :])
# 3d plot
ax.set_xlabel('displacement')
ax.set_ylabel('horsepower')
ax.set_zlabel('mpg')
ax.grid(True)
ax.legend(loc='upper left')
plt.title("linear regression")
range_horsepower = np.arange(50, 225, 30)
range_displacement = np.linspace(100, 500, 10)
mesh_displacement, mesh_horsepower = np.meshgrid(range_displacement, range_horsepower)
mesh_mpg = omega[1] * mesh_displacement + omega[2] * mesh_horsepower + omega[0]
ax.plot_wireframe(mesh_displacement, mesh_horsepower, mesh_mpg)
plt.show()
