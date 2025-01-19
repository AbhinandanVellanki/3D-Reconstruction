import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate, triangulate_3
# Insert your package here

"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=300):
    # pts = [x, y, confidence]

    # Extract the 2D points and confidences
    x1,y1,conf1 = pts1[:,0], pts1[:,1], pts1[:,2]
    x2,y2,conf2 = pts2[:,0], pts2[:,1], pts2[:,2]
    x3,y3,conf3 = pts3[:,0], pts3[:,1], pts3[:,2]

    # Loop through correspondences
    P = []
    err = 0
    for i in range(pts1.shape[0]):
        # Check if confidence is above threshold
        if conf1[i] > Thres and conf2[i] > Thres and conf3[i] > Thres:
            # 3view triangulation
            # Get 2D points
            pt1 = np.array([[x1[i], y1[i]]])
            pt2 = np.array([[x2[i], y2[i]]])
            pt3 = np.array([[x3[i], y3[i]]])

            # Triangulate the 3D point
            p, error = triangulate_3(C1, pt1, C2, pt2, C3, pt3)
            P.append(p)
            err += error
        elif conf1[i] > Thres and conf2[i] > Thres and conf3[i] <= Thres:
            # 2view triangulation
            # Get 2D points
            pt1 = np.array([[x1[i], y1[i]]])
            pt2 = np.array([[x2[i], y2[i]]])

            # Triangulate the 3D point
            p, error = triangulate(C1, pt1, C2, pt2)
            P.append(p)
            err += error
        elif conf1[i] > Thres and conf2[i] <= Thres and conf3[i] > Thres:
            # 2view triangulation
            # Get 2D points
            pt1 = np.array([[x1[i], y1[i]]])
            pt3 = np.array([[x3[i], y3[i]]])

            # Triangulate the 3D point
            p, error = triangulate(C1, pt1, C3, pt3)
            P.append(p)
            err += error
        elif conf1[i] <= Thres and conf2[i] > Thres and conf3[i] > Thres:
            # 2view triangulation
            # Get 2D points
            pt2 = np.array([[x2[i], y2[i]]])
            pt3 = np.array([[x3[i], y3[i]]])

            # Triangulate the 3D point
            p, error = triangulate(C2, pt2, C3, pt3)
            P.append(p)
            err += error
        else:
            print("cannot triangulate at:", i)
            continue
    
    P = np.array(P)
    # remove redundant dimension
    P = np.squeeze(P, axis=1)

    return P, err


"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    
    # Define figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Plot 3D points over time
    for i in range(len(pts_3d_video)):
        pt_3D = pts_3d_video[i]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pt_3D[index0, 0], pt_3D[index1, 0]]
            yline = [pt_3D[index0, 1], pt_3D[index1, 1]]
            zline = [pt_3D[index0, 2], pt_3D[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j], alpha=i / len(pts_3d_video))
    plt.show()


# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    min_err = 1e9
    best_P = None
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        # img_2 = visualize_keypoints(im2, pts2)
        # img_1 = visualize_keypoints(im1, pts1)
        # img_3 = visualize_keypoints(im3, pts3)

        # Get 3D points
        P, err = MultiviewReconstruction(K1@M1, pts1, K2@M2, pts2, K3@M3, pts3, Thres=300)
        # Plot 3D points
        #plot_3d_keypoint(P)
        
        if err < min_err:
            min_err = err
            best_P = P
    
        # Append 3D points to list
        pts_3d_video.append(P)

    # Save the best 3D points
    np.savez("q6_1.npz", P=best_P)

    # Plot 3D points over time
    # plot_3d_keypoint_video(pts_3d_video)
