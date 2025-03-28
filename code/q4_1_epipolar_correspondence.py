import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        print("Finding match for:", xc, yc)
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        plt.draw()


"""
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

"""

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # number of pixels surrounding the point of interest in the search window
    window_size = 2
    # distance threshold for the best match
    dist_threshold = 50

    # 1. Get the epipolar line in im2
    v = np.array([x1, y1, 1])
    l = F @ v

    # 2. Get search window from im1
    search_window = im1[
        y1 - window_size : y1 + window_size + 1, x1 - window_size : x1 + window_size + 1
    ]

    # Generate Gaussian kernel
    kernel_size= 2 * window_size + 1
    sigma = 0.5
    coords = np.linspace(-1, 1, kernel_size)
    y, x = np.meshgrid(coords, coords)
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    kernel /= kernel.sum()

    # 3. Search along the epipolar line in im2 for best match
    least_error = float("inf")
    best_x2, best_y2 = 0, 0

    for y2 in range(im2.shape[0]):
        x2 = int(-1 * (l[1] * y2 + l[2]) / l[0])
        if x2 < 0 or x2 >= im2.shape[1]:
            continue

        # Get search window from im2
        search_window2 = im2[
            y2 - window_size : y2 + window_size + 1, x2 - window_size : x2 + window_size + 1
        ]

        if search_window2.shape != search_window.shape:
            continue

        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > dist_threshold:
            continue

        # Apply Gaussian weighting to each channel of the RGB difference image and compute the difference
        error = 0
        for channel in range(3):
            diff = np.absolute(search_window[:, :, channel] - search_window2[:, :, channel])
            weighted_diff = np.multiply(diff, kernel) # apply gaussian kernel
            error += np.linalg.norm(weighted_diff)
        error /= (2 * window_size + 1) ** 2 # normalize the error

        # Update the best match
        if error < least_error:
            least_error = error
            best_x2, best_y2 = x2, y2

    return best_x2, best_y2


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    print(im1.shape, im2.shape)

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    np.savez("q4_1.npz", F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10
