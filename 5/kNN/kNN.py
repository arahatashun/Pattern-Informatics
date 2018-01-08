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
        modal  = np.argmax(summation, axis = 1)

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
        print("y.dtype", y.dtype)
        print("y.shape", y.shape)
        print(type(y))
        t = np.argmax(t, axis = 1)
        print("t",t)
        print("t.dtype", t.dtype)
        print("t.shape", t.shape)
        print(type(t))
        print(np.equal(y,t))
        correct = np.sum(y == t)
        print("y[1]", y[1])
        print("t[1]", t[1])
        print("correct num",correct)
        accuracy = correct / float(x.shape[0])
        return accuracy



def main(k):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = kNN(x_train, t_train, k)
    print(x_test.shape)
    batch_size =11
    iterations = int(t_test.shape[0]/batch_size)
    for i in range(iterations):
        x_test_batch = x_test[i*batch_size:(i+1)*batch_size]
        t_test_batch = t_test[i*batch_size:(i+1)*batch_size]
        print("batch shape",x_test_batch.shape)
        x_test_batch = np.reshape(x_test_batch, (x_test_batch.shape[0], 1, x_test_batch.shape[1]))
        print("batch shape",x_test_batch.shape)
        rate = network.accuracy(x_test_batch, t_test_batch)
        print("rate", rate)

if __name__ == '__main__':
    k = int(input("Input k"))
    start = time.time()
    main(k)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
