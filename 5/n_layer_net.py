# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""3 layer neural network.

実装にあたっては『ゼロから作る Deep Learning』及びそのリポジトリ
https://github.com/oreilly-japan/deep-learning-from-scratchの
TwoLayerNetを参考にした.
"""

import numpy as np
from functions import sigmoid, softmax, cross_entropy_error
from gradient import numerical_gradient

class NLayerNet:

    def __init__(self, layer_num, input_size, output_size, hidden_size = 100, weight_init_std = 0.01):
        # 重みの初期化
        self.layer_num =layer_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_init_std = weight_init_std
        self.weights = []
        self._make_weights()
        self.bias = []
        self._make_bias()


    def _make_weights(self):
        """
        make wights list
        """
        for i in range(self.layer_num):
            if i == 0:
                #input -> hidden
                self.weights.append(self.weight_init_std * np.random.randn(self.input_size, self.hidden_size))
            elif i == self.layer_num -1:
                #hidden -> output
                self.weights.append(self.weight_init_std * np.random.randn(self.hidden_size, self.output_size))
            else:
                #hidden -> hidden
                self.weights.append(self.weight_init_std * np.random.randn(self.hidden_size, self.hidden_size))

    def _make_bias(self):
        """
        make bias list
        """
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                self.bias.append(np.zeros(self.output_size))
            else:
                self.bias.append(np.zeros(self.hidden_size))

    def predict(self, x):
        """ predict.
        :param x:input_size matrix
        :return: output_size matrix
        """
        input = x
        for i in range(self.layer_num):
            output = np.dot(input, self.weights[i]) + self.bias[i]
            input = sigmoid(output)

        output = np.dot(input, self.weights[-1])+self.bias[-1]
        y = softmax(ouput)
        return y

    def loss(self, x, t):
        """損失関数の値を求める

        :param x: 画像データ
        :param t:正解ラベル
        :return:損失関数
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """認識精度を求める.

        :param x: input data
        :param t: label
        :return:認識精度
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        """重みパラメータに対する勾配

        :param x:input data
        :param t:label
        :return weight_grads:list
        :return bias_grads:list
        """
        loss_W = lambda W: self.loss(x, t)
        weight_grads = []
        bias_grads = []
        for i in range(self.layer_num):
            weight_grads.append(numerical_gradient(loss_W, self.weights[i]))
            bias_grads.append(numerical_gradient(loss_W, self.bias[i]))

        return weight_grads, bias_grads
