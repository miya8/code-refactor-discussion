#!/usr/bin/env python
# coding: utf-8

import keras.utils
import keras.models
import keras.layers.core
import keras.layers
import keras.datasets
import numpy as np
import matplotlib.pyplot as plt

RGB_NUM = 255
HIDDEN_DENCE_NUM = 512
OUTPUT_DENCE_NUM = 10
PIXCEL_SIZE = 784
EPOCH_NUM = 10
BATCH_SIZE = 200
MODEL_VERBOSE = 1
EVALUATE_VERBOSE = 1

# TODO: create_model関数のパラメータの持ち方を改善し、以下リストの重複をなんとかする
param_list = [
    [HIDDEN_DENCE_NUM, 'sigmoid', PIXCEL_SIZE, OUTPUT_DENCE_NUM, 'softmax',
     'categorical_crossentropy', 'sgd', ['accuracy'], BATCH_SIZE, MODEL_VERBOSE, EPOCH_NUM],
    [HIDDEN_DENCE_NUM, 'relu', PIXCEL_SIZE, OUTPUT_DENCE_NUM, 'softmax',
     'categorical_crossentropy', 'sgd', ['accuracy'], BATCH_SIZE, MODEL_VERBOSE, EPOCH_NUM],
    [HIDDEN_DENCE_NUM, 'relu', PIXCEL_SIZE, OUTPUT_DENCE_NUM, 'softmax',
     'categorical_crossentropy', 'adam', ['accuracy'], BATCH_SIZE, MODEL_VERBOSE, EPOCH_NUM],
    [HIDDEN_DENCE_NUM, 'relu', PIXCEL_SIZE, OUTPUT_DENCE_NUM, 'softmax',
     'categorical_crossentropy', 'adam', ['accuracy'], BATCH_SIZE, MODEL_VERBOSE, EPOCH_NUM, 0.2]
]

def create_model(hid_dence_num, hid_activation, pixcel_size, output_dence_num, output_activation, loss, optimizer, metrics, drop_rate=None):
    '''モデルを生成する'''

    model = keras.models.Sequential()
    # Dense = 層 activation = 活性化関数
    # 隠れ層
    model.add(keras.layers.Dense(hid_dence_num,
                                 activation=hid_activation, input_shape=(pixcel_size,)))
    if not drop_rate is None:
        model.add(keras.layers.core.Dropout(drop_rate))
    # 出力層(いくつかのカテゴライズを行う場合はsoftmaxを使う)
    model.add(keras.layers.Dense(
        output_dence_num, activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def plot(X_train, y_train):
    '''画像を表示する'''

    plt.style.use('ggplot')
    idx = 0
    size = 28
    a, b = np.meshgrid(range(size), range(size))
    c = X_train[idx].reshape(size, size)
    c = c[::-1, :]
    print('描かれている数字: {}'.format(y_train[idx]))
    plt.figure(figsize=(2.5, 2.5))
    plt.xlim(0, 27)
    plt.ylim(0, 27)
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    plt.pcolor(a, b, c)
    plt.gray()


def main():

    # y_trainが正解タグデータ
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # ダミーコーディング
    y_train = keras.utils.np_utils.to_categorical(y_train)
    y_test = keras.utils.np_utils.to_categorical(y_test)
    # データの整形&正規化
    X_train = X_train.reshape(60000, PIXCEL_SIZE) / RGB_NUM
    X_test = X_test.reshape(10000, PIXCEL_SIZE) / RGB_NUM

    score_dict = {}
    for no, param in enumerate(param_list):
        model = create_model(*param)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                  verbose=MODEL_VERBOSE, epochs=EPOCH_NUM)
        score_dict[no] = model.evaluate(
            X_test, y_test, verbose=EVALUATE_VERBOSE)
