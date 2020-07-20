import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import random
import main_functions as main


# get mini batches from the training data
def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    data_row = im_train.shape[0]
    data_col = im_train.shape[1]
    rd_size = int(data_col / batch_size)
    mini_batch_x = np.zeros((rd_size, data_row, batch_size))
    mini_batch_y = np.zeros((rd_size, 10, batch_size))
    rd_seed = np.arange(rd_size)
    # In this case rd_seed = range(375), mini batch will be 375 * 196 * 32, In the nth Iteration,
    # I shuffle the rd_seed and put im_train[375 * n + rd_seed[j]] into the nth element in every mini_batch
    for i in range(batch_size):
        np.random.shuffle(rd_seed)
        for j in range(rd_size):
            mini_batch_x[j, :, i] = im_train[:, i * rd_size + rd_seed[j]]
            mini_batch_y[j, label_train[:, i * rd_size + rd_seed[j]], i] = 1
    return mini_batch_x, mini_batch_y


# full connect layer, y=w*x + b
def fc(x, w, b):
    # TO DO
    y = np.dot(w, x) + b
    return y


# full connect layer back propagation
def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dx = np.dot(dl_dy, w)
    dl_dw = np.transpose(np.dot(x, dl_dy))
    dl_db = np.transpose(dl_dy)
    return dl_dx, dl_dw, dl_db


# Euclidean loss (y - y_pred)^2
def loss_euclidean(y_tilde, y):
    # TO DO
    l = np.sum((y - y_tilde) * (y - y_tilde))
    dl_dy = 2 * (y_tilde - y)
    return l, dl_dy


# cross entropy, loss = -y*ln(y_pred)
def loss_cross_entropy_softmax(x, y):
    # TO DO
    x_exp = np.exp(x)
    y_pred = x_exp / np.sum(x_exp)
    l = -np.sum(y * np.log(y_pred))
    dl_dy = y_pred - y
    return l, dl_dy


# leaky relu here, e=0.01
def relu(x):
    # TO DO
    e = 0.01
    y = x * (x >= 0) + e * x * (x < 0)
    return y


# leaky relu back propagation, y>=0, f'=1, else f'=e
def relu_backward(dl_dy, x, y):
    # TO DO
    e = 0.01
    dl_dx = dl_dy * (y >= 0) + e * dl_dy * (y < 0)
    return dl_dx


# im2col function, to unfold the origin images into a convolutional shape
def im2col(im, f_h, f_w, stride=1, pad=1):
    h, w, layer = im.shape
    out_h = f_h * f_w
    out_w = h * w
    im_col = np.zeros((out_h, out_w, layer))
    conv_ctr = int(f_h / 2)
    for l in range(layer):
        for i in range(conv_ctr - pad, h - conv_ctr + pad, stride):
            for j in range(conv_ctr - pad, w - conv_ctr + pad, stride):
                for f_i in range(f_h):
                    for f_j in range(f_w):
                        pt_h = i + f_i - conv_ctr
                        pt_w = j + f_j - conv_ctr
                        if pt_h < 0 or pt_w < 0 or pt_h > h - 1 or pt_w > w - 1:
                            im_col[f_i * f_w + f_j, i * w + j, l] = 0
                        else:
                            im_col[f_i * f_w + f_j, i * w + j, l] = im[pt_h, pt_w, l]
    return im_col


# im2col the input images, and convolute them with w, here I reshape the w directly as we only have one layer image
def conv(x, w_conv, b_conv):
    # TO DO
    w_conv_h, w_conv_w, input_layer, output_layer = w_conv.shape
    x_h, x_w, input_layer = x.shape
    conv_kern_size = int(math.sqrt(w_conv_w * w_conv_h))
    x_new = im2col(x, conv_kern_size, conv_kern_size)
    y_col = np.reshape(w_conv, (output_layer, conv_kern_size * conv_kern_size)).dot(x_new[:, :, 0])+ b_conv
    y = np.zeros((x_h, x_w, output_layer))
    for i in range(output_layer):
        y[:, :, i] = y_col[i].reshape((x_h, x_w))
    return y


# the back propagation of convolution
def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    w_conv_h, w_conv_w, input_layer, output_layer = w_conv.shape
    conv_kern_size = int(math.sqrt(w_conv_w * w_conv_h))
    x_new = im2col(x, conv_kern_size, conv_kern_size)
    dl_dy_col = np.zeros((output_layer, x_new.shape[1]))
    for i in range(output_layer):
        dl_dy_col[i] = dl_dy[:, :, i].reshape((-1))
    dl_dw_col = dl_dy_col.dot(x_new[:, :, 0].T)
    dl_dw = dl_dw_col.reshape(w_conv.shape)
    dl_db = dl_dy_col.sum(axis=1).reshape((3, 1))
    return dl_dw, dl_db


# 2x2 window and stride 2 to pool the data
def pool2x2(x):
    # TO DO
    x_h, x_w, layer = x.shape
    y_h, y_w = int(x_h / 2), int(x_w / 2)
    y = np.zeros((y_h, y_w, layer))
    for l in range(layer):
        for i in range(y_h):
            for j in range(y_w):
                y[i, j, l] = max([x[2 * i, 2 * j, l], x[2 * i, 2 * j + 1, l],
                                  x[2 * i + 1, 2 * j, l], x[2 * i + 1, 2 * j + 1, l]])
    return y


# put data back to the original place
def pool2x2_backward(dl_dy, x, y):
    # TO DO
    dl_dy_h, dl_dy_w, dl_dy_layer = dl_dy.shape
    dl_dx = np.zeros((dl_dy_h * 2, dl_dy_w * 2, dl_dy_layer))
    for i in range(dl_dy_layer):
        for j in range(dl_dy_h):
            for k in range(dl_dy_w):
                if x[2 * j, 2 * k, i] == y[j, k, i]:
                    dl_dx[2 * j, 2 * k, i] = dl_dy[j, k, i]
                elif x[2 * j, 2 * k + 1, i] == y[j, k, i]:
                    dl_dx[2 * j, 2 * k + 1, i] = dl_dy[j, k, i]
                elif x[2 * j + 1, 2 * k, i] == y[j, k, i]:
                    dl_dx[2 * j + 1, 2 * k, i] = dl_dy[j, k, i]
                else:
                    dl_dx[2 * j + 1, 2 * k + 1, i] = dl_dy[j, k, i]
    return dl_dx


# make the matrix a vector
def flattening(x):
    # TO DO
    x_h, x_w, layer = x.shape
    y = np.zeros((x_h * x_w * layer, 1))
    flat_start = 0
    flat_len = x_h * x_w
    for i in range(layer):
        y[flat_start: flat_start + flat_len] = np.reshape(x[:, :, i], (flat_len, 1), 'F')
        flat_start += flat_len
    return y


# make the vector back into matrix
def flattening_backward(dl_dy, x, y):
    # TO DO
    x_h, x_w, layer = x.shape
    dl_dx = np.zeros(x.shape)
    flat_start = 0
    flat_len = x_h * x_w
    for i in range(layer):
        dl_dx[:, :, i] = np.reshape(dl_dy[flat_start: flat_start + flat_len], (x_h, x_w), 'F')
        flat_start += flat_len
    return dl_dx


# train single layer linear perceptron
def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    l_r = 0.016
    d_r = 0.5
    w = np.random.normal(0, 1, (mini_batch_y.shape[1], mini_batch_x.shape[1]))
    b = np.zeros((mini_batch_y.shape[1], 1))
    mini_batch_num = mini_batch_x.shape[0]
    mini_batch_data_len = mini_batch_x.shape[1]
    mini_batch_label_len = mini_batch_y.shape[1]
    mini_batch_size = mini_batch_x.shape[2]
    k = 0
    for i in range(1, 5000):
        if i % 1000 == 0:
            l_r = l_r * d_r
        dl_dw = dl_db = 0
        for j in range(mini_batch_size):
            cur_batch_x = np.reshape(mini_batch_x[k, :, j], (mini_batch_data_len, 1))
            cur_batch_y = np.reshape(mini_batch_y[k, :, j], (mini_batch_label_len, 1))
            y_pred = fc(cur_batch_x, w, b)
            y_pred = np.transpose(y_pred)
            l, dl_dy = loss_euclidean(y_pred, np.transpose(cur_batch_y))
            dl_dx, dl_dw_tmp, dl_db_tmp = fc_backward(dl_dy, cur_batch_x, w, b, cur_batch_y)
            dl_dw += dl_dw_tmp
            dl_db += dl_db_tmp
            # print(l)
        k += 1
        if k >= mini_batch_num:
            k = 0
        w = w - l_r * dl_dw / mini_batch_size
        b = b - l_r * dl_db / mini_batch_size
    return w, b


# train single layer perceptron
def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    l_r = 0.32
    d_r = 0.5
    w = np.random.normal(0, 1, (mini_batch_y.shape[1], mini_batch_x.shape[1]))
    b = np.zeros((mini_batch_y.shape[1], 1))
    mini_batch_num = mini_batch_x.shape[0]
    mini_batch_data_len = mini_batch_x.shape[1]
    mini_batch_label_len = mini_batch_y.shape[1]
    mini_batch_size = mini_batch_x.shape[2]
    k = 0
    for i in range(1, 5000):
        if i % 1000 == 0:
            l_r = l_r * d_r
        dl_dw = dl_db = 0
        for j in range(mini_batch_size):
            cur_batch_x = np.reshape(mini_batch_x[k, :, j], (mini_batch_data_len, 1))
            cur_batch_y = np.reshape(mini_batch_y[k, :, j], (mini_batch_label_len, 1))
            y_pred = fc(cur_batch_x, w, b)
            y_pred = np.transpose(y_pred)
            l, dl_dy = loss_cross_entropy_softmax(y_pred, np.transpose(cur_batch_y))
            dl_dx, dl_dw_tmp, dl_db_tmp = fc_backward(dl_dy, cur_batch_x, w, b, cur_batch_y)
            dl_dw += dl_dw_tmp
            dl_db += dl_db_tmp
        k += 1
        if k >= mini_batch_num:
            k = 0
        w = w - l_r * dl_dw / mini_batch_size
        b = b - l_r * dl_db / mini_batch_size
    return w, b


# train multi-layer perceptron
def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    l_r = 0.64
    d_r = 0.9
    mini_batch_num = mini_batch_x.shape[0]
    mini_batch_data_len = mini_batch_x.shape[1]
    mini_batch_label_len = mini_batch_y.shape[1]
    mini_batch_size = mini_batch_x.shape[2]
    w1 = np.random.normal(0, 1, (30, mini_batch_data_len))
    b1 = np.zeros((30, 1))
    w2 = np.random.normal(0, 1, (mini_batch_label_len, 30))
    b2 = np.zeros((mini_batch_label_len, 1))
    k = 0
    for i in range(1, 5000):
        if i % 1000 == 0:
            l_r = l_r * d_r
        l_iter = 0
        dl_dw1 = dl_db1 = dl_dw2 = dl_db2 = 0
        for j in range(mini_batch_size):
            cur_batch_x = np.reshape(mini_batch_x[k, :, j], (mini_batch_data_len, 1))
            cur_batch_y = np.reshape(mini_batch_y[k, :, j], (mini_batch_label_len, 1))
            y_pred_layer1 = fc(cur_batch_x, w1, b1)
            x_layer2 = relu(y_pred_layer1)
            y_pred_layer2 = fc(x_layer2, w2, b2)
            y_pred_layer2 = np.transpose(y_pred_layer2)
            l, dl_dy_layer2 = loss_cross_entropy_softmax(y_pred_layer2, np.transpose(cur_batch_y))
            l_iter += l
            dl_dx_layer2, dl_dw2_tmp, dl_db2_tmp = fc_backward(dl_dy_layer2, x_layer2, w2, b2, cur_batch_y)
            dl_dw2 += dl_dw2_tmp
            dl_db2 += dl_db2_tmp

            dl_dy_pred_layer1 = relu_backward(dl_dx_layer2, y_pred_layer1, x_layer2.T)

            dl_dx_layer1, dl_dw1_tmp, dl_db1_tmp = fc_backward(dl_dy_pred_layer1, cur_batch_x, w1, b1, y_pred_layer1)
            dl_dw1 += dl_dw1_tmp
            dl_db1 += dl_db1_tmp
        # print(l_iter)
        k += 1
        if k >= mini_batch_num:
            k = 0
        w1 = w1 - l_r * dl_dw1 / mini_batch_size
        b1 = b1 - l_r * dl_db1 / mini_batch_size

        w2 = w2 - l_r * dl_dw2 / mini_batch_size
        b2 = b2 - l_r * dl_db2 / mini_batch_size
    return w1, b1, w2, b2


# train CNN
def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    l_r = 3.2
    d_r = 0.8
    mini_batch_num = mini_batch_x.shape[0]
    mini_batch_data_len = mini_batch_x.shape[1]
    mini_batch_label_len = mini_batch_y.shape[1]
    mini_batch_size = mini_batch_x.shape[2]
    w_conv = np.random.normal(0, 1, (3, 3, 1, 3))
    b_conv = np.zeros((3, 1))
    w_fc = np.random.normal(0, 1, (mini_batch_label_len, 147))
    b_fc = np.zeros((mini_batch_label_len, 1))
    k = 0
    for i in range(1, 12500):
        if i % 1000 == 0:
            l_r = l_r * d_r
        dl_dw_conv = dl_db_conv = dl_dw_fc = dl_db_fc = 0
        l_iter = 0
        for j in range(mini_batch_size):
            cur_batch_x = np.reshape(mini_batch_x[k, :, j], (14, 14, 1), 'F')
            cur_batch_y = np.reshape(mini_batch_y[k, :, j], (mini_batch_label_len, 1))
            y_conv = conv(cur_batch_x, w_conv, b_conv)
            y_relu = relu(y_conv)
            y_pool = pool2x2(y_relu)
            y_flatten = flattening(y_pool)
            y_fc = fc(y_flatten, w_fc, b_fc)
            y_pred = y_fc.T
            l, dl_dy_pred = loss_cross_entropy_softmax(y_pred, np.transpose(cur_batch_y))
            l_iter += l

            dl_dy_flat, dl_dw_fc_tmp, dl_db_fc_tmp = fc_backward(dl_dy_pred, y_flatten, w_fc, b_fc, cur_batch_y)
            dl_dw_fc += dl_dw_fc_tmp
            dl_db_fc += dl_db_fc_tmp

            dl_dy_pool = flattening_backward(dl_dy_flat.T, y_pool, y_flatten)
            dl_dy_relu = pool2x2_backward(dl_dy_pool, y_relu, y_pool)
            dl_dy_conv = relu_backward(dl_dy_relu, y_conv, y_relu)
            dl_dw_conv_tmp, dl_db_conv_tmp = conv_backward(dl_dy_conv, cur_batch_x, w_conv, b_conv, 0)
            dl_dw_conv += dl_dw_conv_tmp
            dl_db_conv += dl_db_conv_tmp
        k += 1
        if k >= mini_batch_num:
            k = 0
        w_conv -= l_r * dl_dw_conv / mini_batch_size
        b_conv -= l_r * dl_db_conv / mini_batch_size

        w_fc -= l_r * dl_dw_fc / mini_batch_size
        b_fc -= l_r * dl_db_fc / mini_batch_size
        print(l_iter)
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



