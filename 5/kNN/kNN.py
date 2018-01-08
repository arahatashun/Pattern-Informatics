# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""kNN.

https://steven.codes/blog/ml/how-to-get-97-percent-on-MNIST-with-KNN/
を参考にした
"""
import os, sys
os.chdir("../")
sys.path.append(os.pardir)
from mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import time

class kNN:

    def __init__(self, train_data, label, k):
        self.train_data = train_data
        self.label = label
        self.k = k

    def predict(self, test):

        diff = self.train_data - test
        print("diff shape",diff.shape)
        distance = np.linalg.norm(diff, axis = 2)
        print("distance shape", distance.shape)
        nearest_neighborhoods = \
        np.array([self.label[np.argsort(distance[i])[:(self.k)]] for i in range(distance.shape[0])])
        # print(k)
        print("nn shape",nearest_neighborhoods.shape)
        summation = nearest_neighborhoods.sum(axis= 1)
        print("sum", summation)
        print("sum shape",summation.shape)
        modal  = np.argmax(summation, axis = 0)

        print(modal)
        return modal

    def accuracy(self, x, t):
        """認識精度を求める.

        :param x: input data
        :param t: label
        :return:認識精度
        """
        y = self.predict(x)
        print("y",y)
        y = np.argmax(y, axis=0)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy



def main(k):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = kNN(x_train, t_train, k)
    print(x_test.shape)
    x_test = x_test[:9]
    print(x_test.shape)
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    print(x_test.shape)
    rate = network.accuracy(x_test, t_test)
    print(rate)

if __name__ == '__main__':
    k = int(input("Input k"))
    start = time.time()
    main(k)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
