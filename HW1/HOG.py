import cv2
import numpy as np
import matplotlib.pyplot as plt


# get differential filter here, professor request to use Gaussian blur or sobel filter
def get_differential_filter():
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # A simplest sobel filter
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # It is the combination of Gaussian filter and differencial filter
    return filter_x, filter_y


# input filter then filter the image, the filter can be any size
def filter_image(im, filter):
    x, y = im.shape[0: 2]
    x_f, y_f = filter.shape[0: 2]
    half_x_f, half_y_f = int(x_f / 2), int(y_f / 2)
    im_tmp = np.zeros([x + 2 * half_x_f, y + 2 * half_y_f], dtype = float)
    im_tmp[half_x_f: x + half_x_f, half_y_f: y + half_y_f] = im # im_tmp is like im with a circle of zeros around
    im_filter = np.zeros([x, y], dtype= float)
    for r in range(half_x_f, x + 1):
        for c in range(half_y_f, y + 1):
            for r_f in range(x_f):
                for c_f in range(y_f):  # From the left top to right down, calculate with the filter
                    im_filter[r - half_x_f, c - half_y_f] += filter[r_f, c_f] * im_tmp[r - half_x_f + r_f, c - half_y_f + c_f]
    del im_tmp
    return im_filter


# Get image gradient and angle for each pixel
def get_gradient(im_dx, im_dy):
    tmp_x, tmp_y = im_dx.shape[0: 2]
    grad_mag = np.zeros([tmp_x, tmp_y], dtype=float)
    grad_angle = np.zeros([tmp_x, tmp_y], dtype=float)
    for i in range(tmp_x):
        for j in range(tmp_y):
            grad_mag[i, j] = np.sqrt(im_dx[i, j] * im_dx[i, j] + im_dy[i, j] * im_dy[i, j])
            grad_angle[i, j] = np.arctan2(round(im_dy[i, j], 4), round(im_dx[i, j], 4))  # get rid of too small figures like E-17 due to the precision of python
            if grad_angle[i, j] < 0:
                grad_angle[i, j] += np.pi
            if grad_angle[i, j] == np.pi:
                grad_angle[i, j] = 0
    return grad_mag, grad_angle


# calculate the histogram by classify the angles
def build_histogram(grad_mag, grad_angle, cell_size):
    M, N = np.array(grad_mag.shape[0: 2])/cell_size
    tmp_x, tmp_y = np.array([M, N], dtype=int) * cell_size
    ori_histo = np.zeros([int(M), int(N), 6], dtype=float)
    for i in range(tmp_x):
        for j in range(tmp_y):
            angle_range = int((grad_angle[i, j] + (np.pi / 12))/(np.pi / 6))  # To find where the angles belong to, range from 1-6
            if angle_range < 0:
                continue
            angle_range %= 6 # To avoid the range is out of 6
            ori_histo[int(i / cell_size), int(j / cell_size), angle_range] += grad_mag[i, j]
    return ori_histo


# find the block descriptor
def get_block_descriptor(ori_histo, block_size):
    M, N = np.array(ori_histo.shape[0: 2], dtype=int)
    M_normalized = M - block_size + 1
    N_normalized = N - block_size + 1
    ori_histo_normalized = np.zeros([M_normalized, N_normalized, 6 * block_size * block_size], dtype=float)
    for x_cur in range(M_normalized):
        for y_cur in range(N_normalized):
            for x_block in range(block_size):
                for y_block in range(block_size):
                    tmp_block = 0
                    for block_range in range(6):
                        cur_block = ori_histo[x_cur + x_block, y_cur + y_block, block_range]
                        tmp_block += cur_block ** 2
                    tmp_block += 0.001
                    tmp_block = np.sqrt(tmp_block)
                    for block_range in range(6):  # put the normalized data(on the right) into corresponding position in ori_histo_normalized
                        ori_histo_normalized[x_cur, y_cur, block_range + (x_block + y_block * block_size) * 6] += ori_histo[x_cur + x_block, y_cur + y_block, block_range] / tmp_block
    return ori_histo_normalized

# extract hog here
def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    # cv2.imshow('im_dx', im_dx)
    # cv2.waitKey()
    # cv2.imshow('im_dy', im_dy)
    # cv2.waitKey()
    # cv2.imshow('im_grad_mag', grad_mag)
    # cv2.waitKey()
    # cv2.imshow('im_grad_angle', grad_angle)
    # cv2.waitKey()
    ori_histo = build_histogram(grad_mag, grad_angle, 8)
    hog = get_block_descriptor(ori_histo, 2)
    visualize_hog(im, hog, 8, 2)
    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    im = cv2.imread('data_image/test_nongrey.jpg', 0)
    hog = extract_hog(im)


