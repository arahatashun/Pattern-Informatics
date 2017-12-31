# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""main.

実装にあたっては『ゼロから作る Deep Learning』及びそのリポジトリ
https://github.com/oreilly-japan/deep-learning-from-scratchの
train_neuralnet.pyを参考にした.
"""

import os, sys

sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
import matplotlib.pyplot as plt
from mnist import load_mnist
from n_layer_net import NLayerNet


def main(layer_num):
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

    network = NLayerNet(layer_num, input_size = 784, hidden_size = 50, output_size = 10)

    iters_num = 10000  # 繰り返しの回数を適宜設定する
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []


    train_acc_list = []
    test_acc_list = []
    """epoch.

    1epochが学習において訓練データをすべて使いきったことに対応
    """
    iter_per_epoch = max(train_size / batch_size, 1)


    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        weight_grads, bias_grads = network.numerical_gradient(x_batch, t_batch)

        # パラメータの更新
        for i in range(layer_num):
            network.weights -= learning_rate * weight_grads
            network.bias -= learning_rate * bias_grads

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    layer_num = int(input("Input number of layers"))
    main(layer_num)
