import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(
            PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


# we compute the dsift for each image in this function
def compute_dsift(img, stride=1, size=16):
    # To do
    sift = cv2.xfeatures2d.SIFT_create()
    dense_points = []
    h, w = img.shape[0], img.shape[1]
    cur_x = cur_y = size / 2
    while cur_x <= w - 1:
        while cur_y <= h - 1:
            dense_points.append(cv2.KeyPoint(cur_x, cur_y, size))
            cur_y += stride
        cur_x += stride
        cur_y = size / 2

    dense_feature = sift.compute(img, dense_points)

    # here I can test if I have generated keypoints in this function
    # img2_sift = 0
    # img2_sift = cv2.drawKeypoints(img, dense_points, dense_feature[1], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img2_sift)
    # plt.show()

    return dense_feature


# resize the images here
def get_tiny_image(img, output_size):
    # To do
    feature = cv2.resize(img, output_size)
    feature = np.array(feature, dtype = float)
    m = np.mean(feature)
    feature = feature - m
    ul = np.sqrt(np.sum(feature * feature))
    feature = feature / ul
    return feature


# KNN prediction
def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(feature_train, label_train)
    label_test_pred = neigh.predict(feature_test)
    return label_test_pred


# tiny image classification, output_size is (20, 16)
def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    output_size = (20, 16)
    trn_data_len = len(label_train_list)
    rshp_len = output_size[0] * output_size[1]
    trn_tiny_data = np.zeros((trn_data_len, rshp_len))
    for i in range(trn_data_len):
        img = cv2.imread(img_train_list[i], 0)
        img = get_tiny_image(img, output_size)

        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()

        trn_tiny_data[i] = img.reshape((1, rshp_len))

    tst_data_len = len(label_test_list)
    tst_tiny_data = np.zeros((tst_data_len, rshp_len))
    for i in range(tst_data_len):
        img = cv2.imread(img_test_list[i], 0)
        img = get_tiny_image(img, output_size)
        tst_tiny_data[i] = img.reshape((1, rshp_len))

    k = 15
    label_test_pred = predict_knn(trn_tiny_data, label_train_list, tst_tiny_data, k)

    confusion, accuracy = confusion_and_accuracy(label_classes, label_test_list, label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


# Kmeans to find the vacab
def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    KM = KMeans(n_clusters=dic_size, max_iter=1000)
    vocab = KM.fit(dense_feature_list).cluster_centers_
    return vocab


# Nearest neighbor to find the bow
def compute_bow(feature, vocab):
    # To do
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vocab)
    matches = neigh.kneighbors(feature[1])
    bow_feature = np.zeros(len(vocab))
    for i in matches[1]:
        bow_feature[i] += 1
    ul = np.sqrt(np.sum(bow_feature * bow_feature))
    bow_feature = bow_feature / ul
    return bow_feature


# find the confusion matrix and accuracy of the classification
def confusion_and_accuracy(label_classes, label_list, label_pred):
    label_classes_len = len(label_classes)
    label_len = len(label_list)
    confusion = np.zeros((label_classes_len, label_classes_len))
    accuracy = 0
    for i in range(label_len):
        confusion[label_classes.index(label_list[i])][label_classes.index(label_pred[i])] += 1
    for i in range(label_classes_len):
        confusion[i] = confusion[i] / sum(confusion[i])
        accuracy += confusion[i][i]

    accuracy /= label_classes_len

    return confusion, accuracy


# KNN classification with bow
def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    # vocab has been generated in the main function, so we can just load it from the vocab.txt
    # this part of code is to find the bow of training data and test data
    vocab = np.loadtxt('./test_data/vocab.txt')
    trn_data_len = len(label_train_list)
    tst_data_len = len(label_test_list)
    vocab_len = len(vocab)
    trn_data_bow = np.zeros((trn_data_len, vocab_len))
    tst_data_bow = np.zeros((tst_data_len, vocab_len))
    for i in range(trn_data_len):
        img = cv2.imread(img_train_list[i], 0)
        d_feature = compute_dsift(img, 14, 14)
        trn_data_bow[i] = compute_bow(d_feature, vocab)
    for i in range(tst_data_len):
        img = cv2.imread(img_test_list[i], 0)
        d_feature = compute_dsift(img, 14, 14)
        tst_data_bow[i] = compute_bow(d_feature, vocab)
    np.savetxt('./test_data/trn_bow.txt', trn_data_bow)
    np.savetxt('./test_data/tst_bow.txt', tst_data_bow)

    # I have also saved the bow for training data and testing data, both can be loaded directly
    # trn_data_bow = np.loadtxt('./test_data/trn_bow.txt')
    # tst_data_bow = np.loadtxt('./test_data/tst_bow.txt')

    k = 16
    label_test_pred = predict_knn(trn_data_bow, label_train_list, tst_data_bow, k)
    confusion, accuracy = confusion_and_accuracy(label_classes, label_test_list, label_test_pred)

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


# SVM predict function,
def predict_svm(feature_train, label_train, feature_test, n_classes=1):
    # To do
    label_len = len(label_train)
    bi_label_list = np.zeros((15, label_len))
    label_list_tmp = []
    label_num = -1
    for i in range(label_len):
        if label_train[i] not in label_list_tmp:
            label_list_tmp.append(label_train[i])
            label_num += 1
        bi_label_list[label_num][i] = 1

    label_tst_len = feature_test.shape[0]
    bi_label_list_tst = np.zeros((15, label_tst_len))
    for i in range(0, 15):
        cls = LinearSVC(C=1.31)
        cls.fit(feature_train, bi_label_list[i])
        bi_label_list_tst[i] = cls.decision_function(feature_test)

    label_test_pred_fig = bi_label_list_tst.argmax(axis=0)
    label_test_pred = []
    for i in range(label_tst_len):
        label_test_pred.append(label_list_tmp[label_test_pred_fig[i]])
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    # vocab has been generated in the main function, so we can just load it from the vocab.txt
    # this part of code is to find the bow of training data and test data
    vocab = np.loadtxt('./test_data/vocab.txt')
    trn_data_len = len(label_train_list)
    tst_data_len = len(label_test_list)
    vocab_len = len(vocab)
    trn_data_bow = np.zeros((trn_data_len, vocab_len))
    tst_data_bow = np.zeros((tst_data_len, vocab_len))
    for i in range(trn_data_len):
        img = cv2.imread(img_train_list[i], 0)
        d_feature = compute_dsift(img, 14, 14)
        trn_data_bow[i] = compute_bow(d_feature, vocab)
    for i in range(tst_data_len):
        img = cv2.imread(img_test_list[i], 0)
        d_feature = compute_dsift(img, 14, 14)
        tst_data_bow[i] = compute_bow(d_feature, vocab)
    np.savetxt('./test_data/trn_bow.txt', trn_data_bow)
    np.savetxt('./test_data/tst_bow.txt', tst_data_bow)

    # I have also saved the bow for training data and testing data, both can be loaded directly
    # trn_data_bow = np.loadtxt('./test_data/trn_bow.txt')
    # tst_data_bow = np.loadtxt('./test_data/tst_bow.txt')

    label_test_pred = predict_svm(trn_data_bow, label_train_list, tst_data_bow)
    confusion, accuracy = confusion_and_accuracy(label_classes, label_test_list, label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info(
        "./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    # We build the visual dictionary here, and save it into vocab.txt in the test_data
    # dense_feature_list = 0
    # for i in range(len(img_train_list)):
    #     trn_name = img_train_list[i]
    #     img = cv2.imread(trn_name, 0)
    #     dense_feature = compute_dsift(img, 14, 14)
    #     if i != 0:
    #         dense_feature_list = np.append(dense_feature_list, dense_feature[1], axis=0)
    #     else:
    #         dense_feature_list = dense_feature[1]
    #
    # vocab = build_visual_dictionary(dense_feature_list, 50)
    # np.savetxt('./test_data/vocab.txt', vocab)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




