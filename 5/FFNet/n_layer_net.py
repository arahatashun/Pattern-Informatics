# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""3 layer neural network.

実装にあたっては『ゼロから作る Deep Learning』及びそのリポジトリ
https://github.com/oreilly-japan/deep-learning-from-scratch
のTwoLayerNetを参考にした.
"""

import numpy as np
from gradient import numerical_gradient
from collections import OrderedDict
from layers import Affine, Relu, Sigmoid, SoftmaxWithLoss

class NLayerNet:

    def __init__(self, layer_num, input_size, output_size, hidden_size, weight_init_std = 0.01):
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
        self._make_layers()


    def _make_layers(self):
        self.layers = []
        for i in range(self.layer_num):
            self.layers.append(Affine(self.weights[i], self.bias[i]))

            if i == self.layer_num -1:
                pass
            else :
                self.layers.append(Relu())

        self.lastLayer = SoftmaxWithLoss()


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
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数の値を求める

        :param x: 画像データ
        :param t:正解ラベル
        :return:損失関数
        """
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

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

    def gradient(self, x, t):
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # return
        weight_grads = []
        bias_grads = []
        for i in range(self.layer_num):
            Affine_layer = self.layers[2*i]
            weight_grads.append(Affine_layer.dW)
            bias_grads.append(Affine_layer.db)

        return weight_grads, bias_grads
