import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""

import numpy as np


def triangulate_3(C1, pts1, C2, pts2, C3, pts3):
    num_corr = pts1.shape[0]
    err = 0
    P = []

    for i in range(num_corr):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        x3, y3 = pts3[i]

        A = np.array(
            [
                y1 * C1[2, :] - C1[1, :],
                C1[0, :] - x1 * C1[2, :],
                y2 * C2[2, :] - C2[1, :],
                C2[0, :] - x2 * C2[2, :],
                y3 * C3[2, :] - C3[1, :],
                C3[0, :] - x3 * C3[2, :],
            ]
        )

        U, s, V = np.linalg.svd(A)
        X = V[-1]
        X /= X[3]  # De-homogenize

        # Calculate reprojection errors for all views
        proj_pts1_i_homo = C1 @ X
        proj_pts1_i = proj_pts1_i_homo[:2] / proj_pts1_i_homo[2]
        d_1 = np.linalg.norm(pts1[i] - proj_pts1_i) ** 2

        proj_pts2_i_homo = C2 @ X
        proj_pts2_i = proj_pts2_i_homo[:2] / proj_pts2_i_homo[2]
        d_2 = np.linalg.norm(pts2[i] - proj_pts2_i) ** 2

        proj_pts3_i_homo = C3 @ X
        proj_pts3_i = proj_pts3_i_homo[:2] / proj_pts3_i_homo[2]
        d_3 = np.linalg.norm(pts3[i] - proj_pts3_i) ** 2

        err += d_1 + d_2 + d_3
        P.append(X[:3])

    return np.array(P), err


def triangulate(C1, pts1, C2, pts2):
    # get number of correspondences
    num_corr = pts1.shape[0]

    # reprojection error
    err = 0

    # 3D points
    P = []
    # solve for each 3D point
    for i in range(num_corr):
        # get coords
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # setup and solve least squares problem
        A = np.array(
            [
                y1 * C1[2, :] - C1[1, :],
                C1[0, :] - x1 * C1[2, :],
                y2 * C2[2, :] - C2[1, :],
                C2[0, :] - x2 * C2[2, :],
            ]
        )
        U, s, V = np.linalg.svd(A)

        # get 3D point in homogenous coords
        X = V[-1]
        # print(X)

        # caluclate projection error
        # pts1
        proj_pts1_i_homo = C1 @ X
        # de-homogenize
        proj_pts1_i = proj_pts1_i_homo[:2] / proj_pts1_i_homo[2]
        d_1 = np.linalg.norm(pts1[i] - proj_pts1_i) ** 2

        # pts2
        proj_pts2_i_homo = C2 @ X
        # de-homogenize
        proj_pts2_i = proj_pts2_i_homo[:2] / proj_pts2_i_homo[2]
        d_2 = np.linalg.norm(pts2[i] - proj_pts2_i) ** 2

        # sum up error
        err += d_1 + d_2

        # de-homogenize world point
        X = X[:3] / X[3]

        # add to list
        P.append(X)

    return np.array(P), err


"""
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
"""


def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    # 1. Get four possible M2s
    # Get intrinsics
    K1, K2 = intrinsics["K1"], intrinsics["K2"]

    # Get Essential Matrix
    E = essentialMatrix(F=F, K1=K1, K2=K2)
    #print(E)

    # Get M2s
    M2s = camera2(E)
    #print(M2s)

    # Fix M1
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))

    # 2. Check which M2 is correct by checking if all points are in front of both cameras
    for i in range(M2s.shape[2]):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2
        C1 = K1 @ M1
        P, err = triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
        if np.all(P[:, 2] > 0):
            np.savez(filename, M2=M2, C2=C2, P=P)
            return M2, C2, P
    
    return None, None, None


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    #print(F)

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert err < 500
