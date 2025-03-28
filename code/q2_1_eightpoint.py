import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here


"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""


def eightpoint(pts1, pts2, M):
    # img = (col, row)

    # Get number of correspondences
    num_corr = pts1.shape[0]

    if num_corr < 8:
        raise ValueError("At least 8 point correspondences are required")

    # 1. Scale (normalize) the coordinates by largest image dimension
    # Normalize points to range [0, 1]
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])

    # Normalize pts
    pts1_homo = np.hstack((pts1, np.ones((num_corr, 1))))
    pts2_homo = np.hstack((pts2, np.ones((num_corr, 1))))
    norm_pts1 = (T @ pts1_homo.T).T[:, :2]
    norm_pts2 = (T @ pts2_homo.T).T[:, :2]

    # 2. Setup 8-point algo equation
    x1, y1 = norm_pts1[:, 0], norm_pts1[:, 1]
    x2, y2 = norm_pts2[:, 0], norm_pts2[:, 1]
    A = np.vstack((x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, np.ones(x1.shape[0]).T)).T

    # 3. Solve by svd
    U, s, V = np.linalg.svd(A)

    # Reshape into matrix
    F_norm = V[-1].reshape((3, 3))

    # 4. Singularize
    F_norm = _singularize(F_norm)

    # 5. Refine
    F_norm = refineF(F_norm, norm_pts1, norm_pts2)

    # 6. Unscale
    F = T.T @ F_norm @ T

    # make F[2][2] = 1
    F = 1/F[2][2] * F

    return F


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    M = np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M)
    print(F)
    np.savez("q2_1.npz", F=F, M=M )

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    error = np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F))
    print("Mean Epipolar Error:", error)
    assert error < 1
