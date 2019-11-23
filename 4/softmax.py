# coding:utf-8
# Author:Shun Arahata
"""
datasetの取得と加工には
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/dataset/mnist.py
を用いた
"""
import os, sys

sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist


def softmax(x):
    """Softmax Functions.

    (K,N)->(K,N)
    """
    # print("x",x)
    numerator = np.exp(x)
    denominator = np.sum(np.exp(x), axis=0)
    # print("numerator",numerator)
    # print("numerator shape", numerator.shape)
    # print("denominator shape",denominator.shape)
    # print("denominator", denominator)
    value = numerator / denominator
    return value


def large_p(w, x):
    """Calculate Large P.

    :param w:weight vector (D,K) D is dimension of deta
    :param x:data (N,D)
    :return large_p:(N,K)
    """
    # print("w.T",w.T)
    # print("x.T",x.T)
    in2softmax = np.dot(w.T, x.T)
    # print("in2softmax",in2softmax)
    large_p = softmax(in2softmax).T
    return large_p


def make_gradient(w, x, t):
    """gradient W W=W-eta*gradientW.

    :param eta: constant
    :param x:data(N,D)
    :param t:large_T teacher(N,K)
    """
    error = large_p(w, x) - t
    # print("error shape", error.shape) # (N, K)
    grad_w = np.dot(x.T, error)
    # print("error",np.mean(np.abs(error),axis = 0))

    return grad_w


def main():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=True)

    N = x_train.shape[0]
    D = x_train.shape[1]
    K = t_train.shape[1]
    # print("train_shape",t_train.shape)
    # print("x_train",x_train)
    # initialize Parameters
    w = np.random.rand(D, K)
    # print("w", w)
    eta = 0.01
    batchsize = 10
    for _ in range(10):
        for index in range(0, N, batchsize):
            _x = x_train[index:index + batchsize:, :]
            # print("_x",_x.shape)
            _t = t_train[index:index + batchsize, :]
            w -= eta * make_gradient(w, _x, _t)

    p = large_p(w, x_test)
    total_test = x_test.shape[0]
    count = 0
    for i in range(total_test):
        if np.argmax(p[i, :]) == np.argmax(t_test[i, :]):
            count += 1
    print(w)
    print("Percentage", count / total_test * 100)


if __name__ == '__main__':
    main()
