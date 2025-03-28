import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy
import random

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""


def ransacF(pts1, pts2, M, nIters=500, tol=2):
    if pts1.shape == pts2.shape:
        N = pts1.shape[0]
    else:
        return None

    iterations = 0
    best_F = None
    max_inliers = -1
    inliers = np.zeros(N, dtype=bool)

    while iterations < nIters:
        # select four matching pairs randomly
        chosen_points_idx = np.random.choice(len(pts1), 8, replace=False)
        x1 = pts1[chosen_points_idx]
        x2 = pts2[chosen_points_idx]

        # compute F
        this_F = eightpoint(x1, x2, M)

        # get inliers
        pts1_homo, pts2_homo = toHomogenous(pts1), toHomogenous(pts2)
        err = calc_epi_error(pts1_homo, pts2_homo, this_F)
        inliers_indexes = np.where(err < tol)[0]

        # update best inliers
        if len(inliers_indexes) > max_inliers:
            best_F = this_F
            inliers = np.where(err < tol, True, False).reshape(N, 1)
            max_inliers = len(inliers_indexes)

        iterations += 1

    return best_F, inliers


"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.identity(3) # no rotation
    r = r / theta  # Normalize rotation vector
    r_x = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    R = (
        np.cos(theta) * np.identity(3)
        + np.sin(theta) * r_x
        + (1 - np.cos(theta)) * np.outer(r, r)
    )
    return R


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


def invRodrigues(R):
    A = (R - R.T) / 2
    p = np.array([A[2, 1], A[0, 2], A[1, 0]])
    s = np.linalg.norm(p)
    c = (R.trace() - 1) / 2
    if s == 0 and c == 1:
        return np.zeros(3)
    if s == 0 and c == -1:
        v = np.zeros(3)
        for i in range(3):
            if R[i, i] + 1 > 1e-6:
                v[i] = np.sqrt((R[i, i] + 1) / 2)
                break
        return np.pi * v
    theta = np.arctan2(s, c)
    return theta * p / s


"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Extract P, r2, t2 from x
    P = x[: 3 * len(p1)].reshape(-1, 3) #first 3N elements
    r2 = x[3 * len(p1) : 3 * len(p1) + 3] #next 3 elements 
    t2 = x[3 * len(p1) + 3 :] #remaining elements

    R2 = rodrigues(r2) #get rotation matrix

    # compute C1, C2
    C1 = K1 @ M1
    C2 = K2 @ np.hstack((R2, t2.reshape(-1, 1)))

    # project P to get the estimated points
    p1_est_homo = C1 @ np.hstack((P, np.ones((P.shape[0], 1)))).T
    p2_est_homo = C2 @ np.hstack((P, np.ones((P.shape[0], 1)))).T
    p1_est = (p1_est_homo[:2] / p1_est_homo[2]).T  # project to 2D
    p2_est = (p2_est_homo[:2] / p2_est_homo[2]).T  # project to 2D

    residuals1 = p1 - p1_est  # Error for image 1
    residuals2 = p2 - p2_est  # Error for image 2

    # compute residuals
    residuals = np.concatenate((residuals1.flatten(), residuals2.flatten()))
    return residuals


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""


def obj_fun(x, K1, M1, p1, K2, p2):
    return np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x))**2


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # stack P, r2, t2 to create initial vector for optimization
    R2_init, t2_init = M2_init[:, :3], M2_init[:, 3]
    r2_init = invRodrigues(R2_init)
    x_init = np.hstack((P_init.flatten(), r2_init, t2_init))

    # compute initial objective function value
    obj_start = np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x_init))

    # minimize the objective function
    res = scipy.optimize.minimize(
        fun=obj_fun,
        x0=x_init,
        args=(K1,M1,p1,K2,p2,),
        method="Powell",
    )

    # Get optimized vector
    x_opt = res.x

    # extract optimized P, r2, t2
    P_opt = x_opt[: 3 * len(p1)].reshape(-1, 3)
    r2_opt = x_opt[3 * len(p1) : 3 * len(p1) + 3]
    t2_opt = x_opt[3 * len(p1) + 3 :]

    # compute R2 from r2
    R2_opt = rodrigues(r2_opt)

    # compute M2
    M2_opt = np.hstack((R2_opt, t2_opt[:, np.newaxis]))

    # compute final objective function value
    obj_end = np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x_opt))

    return M2_opt, P_opt, obj_start, obj_end

if __name__ == "__main__":
    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """

    # RANSAC
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M, nIters=500, tol=2)

    # Simple Tests to Verify F Implementation
    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2

    #Simple Tests to Verify rodrigues() and invRodrigues() Implementation
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Get inliers
    inlier_pts1 = noisy_pts1[inliers.flatten()]
    inlier_pts2 = noisy_pts2[inliers.flatten()]

    # Find M2
    M2, C2, P = findM2(F, inlier_pts1, inlier_pts2, intrinsics)

    # Bundle Adjustment
    M2_opt, P_opt, obj_start, obj_end = bundleAdjustment(
        K1=K1,
        M1=np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis])),
        p1=inlier_pts1,
        K2=K2,
        M2_init=M2,
        p2=inlier_pts2,
        P_init=P,
    )

    # Plot 3D points before and after bundle adjustment
    print("Final M2: ", M2_opt)
    print("Initial reprojection error: ", np.linalg.norm(obj_start)**2)
    print("Final reprojection error: ", np.linalg.norm(obj_end)**2)
    plot_3D_dual(P, P_opt)
