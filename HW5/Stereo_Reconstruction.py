import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space, svd
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

def find_match(img1, img2):
    # TO DO
    RoD = 0.7
    sift = cv2.xfeatures2d.SIFT_create()
    kp_img1, des_img1 = sift.detectAndCompute(img1, None)
    kp_img2, des_img2 = sift.detectAndCompute(img2, None)
    x1 = []
    x2 = []
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des_img2)
    matches = neigh.kneighbors(des_img1)
    for i in range(len(kp_img1)):
        if matches[0][i][0] < RoD * matches[0][i][1]:# when d1/d2 < 0.7, it is a good match
            x1.append(kp_img1[i].pt)
            x2.append(kp_img2[matches[1][i][0]].pt)
    # exchange the test case and the training case
    x1_rev = []
    x2_rev = []
    neigh.fit(des_img1)
    matches = neigh.kneighbors(des_img2)
    for i in range(len(kp_img2)):
        if matches[0][i][0] < RoD * matches[0][i][1]:
            x2_rev.append(kp_img2[i].pt)
            x1_rev.append(kp_img1[matches[1][i][0]].pt)
    pts1 = []
    pts2 = []
    for i in range(len(x1)):
        for j in range(len(x1_rev)):
            if x1[i] == x1_rev[j] and x2[i] == x2_rev[j]:
                if x1[i] not in pts1:
                    pts1.append(x1[i])
                    pts2.append(x2[i])
                break
    pts1, pts2 = np.array(pts1), np.array(pts2)
    return pts1, pts2


def compute_F(pts1, pts2):
    # TO DO
    max_inlier = -1
    uv_cur = np.ones((8, 9))
    ransac_iter = 5000
    ransac_thr = 0.01
    F_max = 0
    for t in range(ransac_iter):
        rd_pt = np.random.choice(pts1.shape[0], 8, replace=False)
        cur = 0
        for i in rd_pt:
            ux, uy, vx, vy = pts1[i, 0], pts1[i, 1], pts2[i, 0], pts2[i, 1]
            uv_cur[cur, 0] = ux * vx
            uv_cur[cur, 1] = uy * vx
            uv_cur[cur, 2] = vx
            uv_cur[cur, 3] = ux * vy
            uv_cur[cur, 4] = uy * vy
            uv_cur[cur, 5] = vy
            uv_cur[cur, 6] = ux
            uv_cur[cur, 7] = uy
            cur += 1
        F_vec = null_space(uv_cur)
        F_cur = F_vec.reshape((3, 3))
        U, s, Vh = svd(F_cur)
        sigma = np.zeros((3, 3))
        for i in range(2):
            sigma[i, i] = s[i]
        F_rank2 = np.dot(U, np.dot(sigma, Vh))
        num_inlier = 0
        for i in range(pts1.shape[0]):
            u_tmp = np.ones((3, 1))
            v_trans_tmp = np.ones((1, 3))
            u_tmp[0, 0], u_tmp[1, 0], v_trans_tmp[0, 0], v_trans_tmp[0, 1] = pts1[i][0], pts1[i][1], pts2[i][0], pts2[i][1]
            res_pt = abs(np.dot(v_trans_tmp, np.dot(F_rank2, u_tmp)))
            if res_pt < ransac_thr:
                num_inlier = num_inlier + 1
        if num_inlier > max_inlier:
            max_inlier = num_inlier
            F_max = F_rank2
    F = F_max
    return F


def triangulation(P1, P2, pts1, pts2):
    # TO DO
    n = len(pts1)
    pts3D = np.zeros((n, 3))
    for i in range(n):
        ux, uy, vx, vy = pts1[i, 0], pts1[i, 1], pts2[i, 0], pts2[i, 1]
        A1 = np.array([[0, -1, uy], [1, 0, -ux]]).dot(P1)
        A2 = np.array([[0, -1, vy], [1, 0, -vx]]).dot(P2)
        A = np.vstack((A1, A2))
        U, s, Vh = svd(A)
        V = Vh.T
        X_1 = V[:, 3] / V[3, 3]
        # bb = A.dot(X_1)
        pts3D[i, :] = X_1[0: 3]
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO
    max_cnt = 0
    R_max = C_max = pts3D_max = 0
    for i in range(4):
        R_cur, C_cur, pts3D_cur = Rs[i], Cs[i], pts3Ds[i]
        X_C = pts3D_cur.T - C_cur
        r3 = R_cur[2]
        Cheirality_cond = r3.dot(X_C)
        pstv_cnt = len(Cheirality_cond[Cheirality_cond > 0])
        if pstv_cnt > max_cnt:
            max_cnt = pstv_cnt
            R_max, C_max, pts3D_max = R_cur, C_cur, pts3D_cur
    R, C, pts3D = R_max, C_max, pts3D_max
    return R, C, pts3D


def compute_rectification(K, R, C):
    # TO DO
    rx = C / np.linalg.norm(C)
    rz_tilde = np.array([[0, 0, 1]])
    rz_tilde = rz_tilde.T
    rz_up = rz_tilde - rx[2, 0] * rx
    rz = rz_up / np.linalg.norm(rz_up)
    ry = np.cross(rz.T, rx.T)
    R_rect = np.vstack((rx.T, ry, rz.T))
    K_inv = np.linalg.inv(K)
    H1 = K.dot(R_rect.dot(K_inv))
    H2 = K.dot(R_rect.dot(R.T.dot(K_inv)))
    return H1, H2


def dense_match(img1, img2):
    # TO DO
    sift = cv2.xfeatures2d.SIFT_create()
    img_y, img_x = img1.shape
    size_dense = 5
    padding = int(size_dense / 2) + 1
    img1_tmp = np.pad(img1, ((padding, padding), (padding, padding)), 'constant', constant_values=(0, 0))
    img2_tmp = np.pad(img2, ((padding, padding), (padding, padding)), 'constant', constant_values=(0, 0))

    kp1 = [cv2.KeyPoint(x, y, size_dense) for y in range(padding, img_y + padding) for x in range(padding, img_x + padding)]
    kp2 = [cv2.KeyPoint(x, y, size_dense) for y in range(padding, img_y + padding) for x in range(padding, img_x + padding)]

    img1_ds_tmp = sift.compute(img1_tmp, kp1)[1]
    img2_ds_tmp = sift.compute(img2_tmp, kp2)[1]

    img1_ds = img1_ds_tmp.reshape((img_y, img_x, -1))
    img2_ds = img2_ds_tmp.reshape((img_y, img_x, -1))

    disparity = np.zeros_like(img1)

    for i in range(img_y):
        neigh = NearestNeighbors(n_neighbors=1)
        for j in range(img_x):
            neigh.fit(img2_ds[i, 0: j + 1])
            if img1[i, j] == 0:
                break
            tmp_ds = img1_ds[i, j].reshape((1, -1))
            match = neigh.kneighbors(tmp_ds, return_distance=False)
            # print(match)
            disparity[i, j] = abs(match[0, 0] - j)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
