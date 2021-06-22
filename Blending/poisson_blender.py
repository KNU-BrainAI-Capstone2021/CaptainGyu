import scipy.sparse


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    return mat_A


laplacian_matrix(3, 3).todense()

import os
from os import path
import cv2
import numpy as np
import matplotlib.pyplot as plt
% matplotlib
inline

sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
scr_dir = '/content/gdrive/My Drive/poisson'
name_list = os.listdir(scr_dir + "/source")
output_list = os.listdir(scr_dir + "/output")
for i in name_list:
    source = cv2.imread(path.join(scr_dir + "/source", i))
    target = cv2.imread(path.join(scr_dir + "/target", i))
    mask = cv2.imread(path.join(scr_dir + "/mask", i), cv2.IMREAD_GRAYSCALE)
    offset = (0, 0)
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min

    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    source = cv2.warpAffine(source, M, (x_range, y_range))
    mask = mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1
    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()
    from scipy.sparse.linalg import spsolve

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat) * alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        target[y_min:y_max, x_min:x_max, channel] = x
    out_dir = '/content/gdrive/My Drive/poisson'
    cv2.imwrite(path.join(out_dir + "/output", i), target)
    sharpening_out1 = cv2.filter2D(target, -1, sharpening_mask2)
    cv2.imwrite(path.join(out_dir + "/sharp", i), sharpening_out1)

print("done")