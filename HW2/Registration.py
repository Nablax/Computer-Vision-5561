import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


# get differential filter here, professor request to use Gaussian blur or sobel filter
def get_differential_filter():
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # A simplest sobel filter
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # It is the combination of Gaussian filter and differencial filter
    return filter_x, filter_y


# find SIFT matches
def find_match(img1, img2):
    # To do
    img1_sift = 0
    img2_sift = 0
    sift = cv2.xfeatures2d.SIFT_create()
    kp_img1, des_img1 = sift.detectAndCompute(img1, None)
    kp_img2, des_img2 = sift.detectAndCompute(img2, None)
    # visualization of SIFT
    # img1_sift = cv2.drawKeypoints(img1, kp_img1, img1_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2_sift = cv2.drawKeypoints(img2, kp_img2, img2_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    x1 = []
    x2 = []
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des_img2)
    matches = neigh.kneighbors(des_img1)
    for i in range(len(kp_img1)):
        if matches[0][i][0] < 0.7 * matches[0][i][1]:# when d1/d2 < 0.7, it is a good match
            x1.append(kp_img1[i].pt)
            x2.append(kp_img2[matches[1][i][0]].pt)
    # exchange the test case and the training case
    # neigh.fit(des_img1)
    # matches = neigh.kneighbors(des_img2)
    # for i in range(len(kp_img2)):
    #     if matches[0][i][0] < 0.7 * matches[0][i][1]:
    #         x2.append(kp_img2[i].pt)
    #         x1.append(kp_img1[matches[1][i][0]].pt)
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2


# RANSAC to find the best affine transformation
def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    max_inlier = 0
    A_cur = np.zeros((6, 6))
    B_cur = np.zeros((6, 1))
    A_max = np.zeros((3, 3))
    A_max[2][2] = 1
    for t in range(ransac_iter):
        rd_pt = np.random.choice(x1.shape[0], 3, replace=False)
        cur = 0
        for i in rd_pt:
            x1_now = np.append(x1[i], 1)
            A_cur[cur][0: 3] = A_cur[cur + 1][3: 6] = x1_now
            B_cur[cur][0], B_cur[cur + 1][0] = x2[i][0], x2[i][1]
            cur = cur + 2
        x_cur = np.linalg.lstsq(A_cur, B_cur, rcond = None)[0]
        num_inlier = 0
        for i in range(x1.shape[0]):
            u1, v1, u2, v2 = x1[i][0], x1[i][1], x2[i][0], x2[i][1]
            u2_tmp = x_cur[0][0] * u1 + x_cur[1][0] * v1 + x_cur[2][0]
            v2_tmp = x_cur[3][0] * u1 + x_cur[4][0] * v1 + x_cur[5][0]
            res_pt = np.array([u2_tmp, v2_tmp])
            if np.linalg.norm(res_pt - x2[i]) < ransac_thr:
                num_inlier = num_inlier + 1
        if num_inlier > max_inlier:
            max_inlier = num_inlier
            for i in range(3):
                A_max[0][i] = x_cur[i][0]
                A_max[1][i] = x_cur[i + 3][0]
    A = A_max
    return A


# visualize the RANSAC matches
def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)

    corner = np.zeros((4, 2))
    corner[0] = [0, 0]
    corner[1] = [0, img1.shape[0]]
    corner[2] = [img1.shape[1], 0]
    corner[3] = [img1.shape[1], img1.shape[0]]

    x1n = x1
    x2n = x2
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    for i in range(4):
        u1, v1 = corner[i][0], corner[i][1]
        u2_tmp = A[0][0] * u1 + A[0][1] * v1 + A[0][2]
        v2_tmp = A[1][0] * u1 + A[1][1] * v1 + A[1][2]
        u2_tmp = u2_tmp * scale_factor2 + img1_resized.shape[1]
        v2_tmp = v2_tmp * scale_factor2
        plt.plot(u2_tmp, v2_tmp, 'yo')
    for i in range(x1.shape[0]):
        u1, v1, u2, v2 = x1n[i][0], x1n[i][1], x2n[i][0], x2n[i][1]
        u2_tmp = A[0][0] * u1 + A[0][1] * v1 + A[0][2]
        v2_tmp = A[1][0] * u1 + A[1][1] * v1 + A[1][2]
        res_pt = np.array([u2_tmp, v2_tmp])
        if np.linalg.norm(res_pt - x2n[i]) < ransac_thr:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


# warp the image from the template to the target
def warp_image(img, A, output_size):
    # To do
    # warp image from target to template
    # h, w = img.shape
    # img_warped = np.zeros(output_size)
    # pixel_img = np.zeros((3, 1))
    # pixel_img[2, 0] = 1
    # cc = 0
    # for i in range(h):
    #     for j in range(w):
    #         pixel_img[0, 0] = j
    #         pixel_img[1, 0] = i
    #         pixel_trans = np.linalg.lstsq(A, pixel_img)[0]
    #         u1 = math.floor(pixel_trans[0, 0])
    #         v1 = math.floor(pixel_trans[1, 0])
    #         if u1 < 0 or v1 < 0:
    #             continue
    #         if u1 >= output_size[1] or v1 >= output_size[0]:
    #             continue
    #         img_warped[v1, u1] = img[i, j] / 255.0
    #         cc += 1
    # cv2.imshow('', img_warped)
    # cv2.waitKey()
    h = output_size[0]
    w = output_size[1]
    img_warped = np.zeros(output_size)
    pixel_img = np.zeros((3, 1))
    pixel_img[2, 0] = 1
    for i in range(h):
        for j in range(w):
            pixel_img[0, 0] = j
            pixel_img[1, 0] = i
            pixel_trans = A.dot(pixel_img)
            u1 = math.floor(pixel_trans[0, 0])
            v1 = math.floor(pixel_trans[1, 0])
            img_warped[i, j] = img[v1, u1]
    # plt.imshow(img_warped, cmap='gray')
    # plt.axis('off')
    # plt.show()
    return img_warped


# Inverse Compositional Image Alignment algorithm
def align_image(template, target, A):
    # To do
    P = A
    im_tmp = template.astype('float') / 255.0
    im_tar = target.astype('float') / 255.0
    filter_x, filter_y = get_differential_filter()
    x, y = template.shape[0: 2]
    x_f, y_f = filter_x.shape[0: 2]
    half_x_f, half_y_f = int(x_f / 2), int(y_f / 2)
    im_var = np.zeros([x + 2 * half_x_f, y + 2 * half_y_f], dtype = float)
    im_var[half_x_f: x + half_x_f, half_y_f: y + half_y_f] = im_tmp
    im_filter = np.zeros([x, y, 2], dtype= float)
    im_jac = np.zeros([2, 6], dtype= float)
    im_st_dec = np.zeros([x * y, 6], dtype= float)
    for r in range(x):
        for c in range(y):
            im_jac[0, 0: 3] = im_jac[1, 3: 6] = [c, r, 1]
            for r_f in range(x_f):
                for c_f in range(y_f):  # From the left top to right down, calculate with the filter
                    im_filter[r, c, 0] += filter_x[r_f, c_f] * im_var[r + r_f, c + c_f]
                    im_filter[r, c, 1] += filter_y[r_f, c_f] * im_var[r + r_f, c + c_f]
            for std_now in range(6): # while filtering compute the jacobian
                im_st_dec[r * y + c, std_now] = im_filter[r, c, 0] * im_jac[0, std_now] + im_filter[r, c, 1] * im_jac[1, std_now]
    # show steepest descent images
    # for std_now in range(6):
    #     plt.imshow(np.resize(im_st_dec[:, std_now], [x, y]), cmap='gray')
    #     plt.axis('off')
    #     plt.show()
    P_del_v = 100
    im_err_v = []
    iter_time = 1000
    while iter_time > 0:
        iter_time -= 1
        img_warped = warp_image(im_tar, P, template.shape)
        im_err = im_tmp - img_warped #depending on the Jacobian
        # plt.imshow(im_err, cmap='gray')
        # plt.axis('off')
        # plt.show()
        im_err = np.resize(im_err, (x * y, 1))
        im_err_v.append(np.linalg.norm(im_err))
        # Hession = im_st_dec.T.dot(im_st_dec)
        # im_err_by_hession = im_st_dec.T.dot(im_err)
        # P_del = np.linalg.inv(Hession).dot(im_err_by_hession)
        P_del = np.linalg.lstsq(im_st_dec, im_err, rcond = None)[0]# this line is same as the previous three lines
        P_del_v = np.linalg.norm(P_del)
        P_del = np.resize(P_del, [3, 3])
        P_del[2, :] = [0, 0, 1]
        P_del[0, 0] += 1
        P_del[1, 1] += 1
        P = P.dot(np.linalg.inv(P_del))
        print(P_del_v, im_err_v[-1])
    A_refined = P
    im_err_v = np.array(im_err_v)
    return A_refined, im_err_v


# It takes 20 min to compute this function
def track_multi_frames(template, img_list):
    # To do
    frame_len = len(img_list)
    A_list = np.zeros([frame_len, 3, 3])
    for i in range(frame_len):
        if i == 0:
            tmp_now = template
            x1, x2 = find_match(tmp_now, img_list[i])
            A = align_image_using_feature(x1, x2, 5, 500)
            A_refined, errors = align_image(tmp_now, img_list[i], A)
        else:
            tmp_now = warp_image(img_list[i - 1], A_list[i - 1], template.shape)
            A_refined, errors = align_image(tmp_now, img_list[i], A_list[i - 1])
        A_list[i] = A_refined
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # test_tmp = cv2.imread('test_img/test_template.png', 0)
    # test_pic = cv2.imread('test_img/test_pic.png', 0)
    # x1, x2 = find_match(test_tmp, test_pic)
    # visualize_find_match(test_tmp, test_pic, x1, x2)
    template = cv2.imread('data_img/Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('data_img/Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    # x1, x2 = find_match(template, target_list[0])

    # visualize_find_match(template, target_list[0], x1, x2)

    # ransac_thr = 5
    # ransac_iter = 500
    # A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    # visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

    # img_warped = warp_image(target_list[0], A, template.shape)

    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    # A_refined, errors = align_image(template, target_list[0], A)
    # visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


