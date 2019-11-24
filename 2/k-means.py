# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

df = pd.read_csv("iris.csv")
data = df.values
numberOfData = data.shape[0]
convergence_threshold = 0.001


def addRand2List(list):
    rand = random.randint(0, numberOfData - 1)
    if rand in list:
        addRand2List(list)
    else:
        list.append(rand)


def getInitalRepresentative(k):
    Rp = []
    Rp_matrix = []
    for i in range(k):
        addRand2List(Rp)
    Rp_matrix.append(data[Rp, :])
    Rp_matrix = np.array(Rp_matrix)[0]
    return Rp_matrix


def getDistance(vector_A, vector_B):
    distance_list = []
    # print(vector_A)
    # print(vector_B)
    for i in range(len(vector_A)-1):
        square = (vector_A[i] - vector_B[i]) ** 2
        distance_list.append(square)
    distance = np.sum(distance_list) ** (1 / 2)
    return distance


def ImprovedInitialSelect(k):
    Rp_matrix = []
    tmp_data = data.copy()
    first = random.randint(0, numberOfData - 1)
    Rp_matrix.append(tmp_data[first, :])
    tmp_data = np.delete(tmp_data, first, axis=0)
    for j in range(k - 1):
        distance = []
        for i in range(tmp_data.shape[0]):
            distance.append(getDistance(data[first], tmp_data[i]))
            if j >= 1:
                for k in range(j):
                    distance[i] = distance[i] + getDistance(Rp_matrix[k + 1], tmp_data[i])

        pattern_index = distance.index(max(distance))
        Rp_matrix.append(tmp_data[pattern_index, :])
        tmp_data = np.delete(tmp_data, pattern_index, axis=0)
    Rp_matrix = np.array(Rp_matrix)
    # print(Rp_matrix)
    return Rp_matrix


# get pattern of specific one point accroding to k representative points
# return  pattern using index of k_list
def getPattern(point, k_list):
    distance = []
    for i in range(k_list.shape[0]):
        distance.append(getDistance(k_list[i], data[point]))
    pattern_index = distance.index(min(distance))
    # return data[k_list[pattern_index]][4]
    return pattern_index


def labeling(data, k_list):
    pattern = []
    for i in range(numberOfData):
        pattern.append(getPattern(i, k_list))
    return pattern


# return 2 dimension array
# add cluster number to data
def clustering(k_list):
    tmp_data = data.copy()
    pattern = np.array([labeling(data, k_list)])
    number_of_k = k_list.shape[0]
    cluster = np.concatenate((tmp_data, pattern.T), axis=1)
    # cluster=cluster[tmp_data[:,-1].argsort()]
    return cluster


def normalize_color(k_list, i):
    i = i + 1
    max_color = 1
    num = k_list.shape[0]
    return [i / num * max_color]


# input data and k_list
# return the index of the center of the cluster
def getCenter(k_list):
    tmp_data = clustering(k_list)
    new_k_list = []
    for k in range(k_list.shape[0]):
        cluster = tmp_data[tmp_data[:, -1] == k]
        num = cluster.shape[0]
        # print(num,k)
        # zero division errro?
        if num == 0:
            new_k_list.append(k_list[k][0:4].tolist())
        else:
            a = np.sum(cluster[:, 0]) / num
            b = np.sum(cluster[:, 1]) / num
            c = np.sum(cluster[:, 2]) / num
            d = np.sum(cluster[:, 3]) / num
            new_k_list.append([a, b, c, d])
    new_k_list = np.array(new_k_list)
    return new_k_list


def calcConvergence(old_list, new_list):
    variation = 0
    for i in range(old_list.shape[0]):
        variation = variation + getDistance(old_list[i], new_list[i])
    return variation


def plotrandom(k_list, ax):
    tmp_data = clustering(k_list)
    hot = plt.get_cmap('hot')
    cNorm = colors.Normalize(vmin=0, vmax=len(k_list))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=hot)
    k_len = k_list.shape[0]

    for k in range(k_len):
        cluster = tmp_data[tmp_data[:, -1] == k]
        x_center = k_list[k][0]
        y_center = k_list[k][1]
        z_center = k_list[k][2]
        x = cluster[:, 0]
        y = cluster[:, 1]
        z = cluster[:, 2]
        k_label = 'group' + str(k + 1)
        ax.scatter(x_center, y_center, z_center, s=40, color=scalarMap.to_rgba(k), label=k_label)
        ax.scatter(x, y, z, s=15, color=scalarMap.to_rgba(k), label=k_label)

    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.set_zlabel('petal length')
    ax.grid(True)
    ax.legend(loc='upper left')
    k_num_title = 'random selection ' + 'k=' + str(k_len)
    ax.set_title(k_num_title)


def plotimproved(k_list, ax):
    tmp_data = clustering(k_list)
    hot = plt.get_cmap('hot')
    cNorm = colors.Normalize(vmin=0, vmax=len(k_list))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=hot)
    k_len = k_list.shape[0]

    for k in range(k_len):
        cluster = tmp_data[tmp_data[:, -1] == k]
        x_center = k_list[k][0]
        y_center = k_list[k][1]
        z_center = k_list[k][2]
        x = cluster[:, 0]
        y = cluster[:, 1]
        z = cluster[:, 2]
        k_label = 'group' + str(k + 1)
        ax.scatter(x_center, y_center, z_center, s=40, color=scalarMap.to_rgba(k), label=k_label)
        ax.scatter(x, y, z, s=15, color=scalarMap.to_rgba(k), label=k_label)

    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.set_zlabel('petal length')
    ax.grid(True)
    ax.legend(loc='upper left')
    k_num_title = 'improved selection ' + 'k=' + str(k_len)
    ax.set_title(k_num_title)


def plotscatter(k):
    """
    old_centre = getCenter(getInitalRepresentative(k))
    new_centre = getCenter(old_centre)
    convergence = calcConvergence(old_centre, new_centre)
    old_centre = new_centre
    convergence_count = 1
    while convergence > convergence_threshold:
        new_centre = getCenter(old_centre)
        convergence = calcConvergence(old_centre, new_centre)
        old_centre = new_centre
        convergence_count = convergence_count + 1

    fig = plt.figure(figsize=(15, 4.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    plotrandom(new_centre, ax1)
    print("収束するまでの回数",convergence_count)
    """
    initial = ImprovedInitialSelect(k)
    print(initial)
    old_centre = getCenter(initial)
    new_centre = getCenter(old_centre)
    convergence = calcConvergence(old_centre, new_centre)
    old_centre = new_centre
    convergence_count = 1
    while convergence > convergence_threshold:
        new_centre = getCenter(old_centre)
        convergence = calcConvergence(old_centre, new_centre)
        old_centre = new_centre
        convergence_count = convergence_count + 1

    print("収束するまでの回数=",convergence_count)
    plotimproved(new_centre, ax2)
    filename = "../picture/" + str(k) + ".png"
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    x = input("Please Enter k: ")
    plotscatter(int(x))
