import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    

    """ YOUR CODE HERE
    """
    points = np.array(
        [[(0 - K[0, 2]) * depth / K[0, 0], (0 - K[1, 2]) * depth / K[1, 1], depth],
         [(width - K[0, 2]) * depth / K[0, 0], (0 - K[1, 2]) * depth / K[1, 1], depth],
         [(0 - K[0, 2]) * depth / K[0, 0], (height - K[1, 2]) * depth / K[1, 1], depth],
         [(width - K[0, 2]) * depth / K[0, 0], (height - K[1, 2]) * depth / K[1, 1], depth]]
    )

    points = (np.linalg.inv(Rt[:, 0:3]) @ (points.T - Rt[:, 3].reshape(3, 1))).T.reshape(2, 2, 3)
    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    h, w = points.shape[0], points.shape[1]
    points = K @ (Rt[:, 0:3] @ points.reshape(-1, 3).T + Rt[:, 3].reshape(3, 1))
    points = points / points[2]
    points = points[0:2].T.reshape(h, w, 2)
    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """
    ref_proj = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1],
    ], dtype=np.float32)

    world_corners = backproject_fn(K_ref, width, height, depth, Rt_ref)
    neighbor_proj = project_fn(K_neighbor, Rt_neighbor, world_corners).reshape(-1, 2)
    neighbor_proj = np.concatenate((neighbor_proj, np.ones((neighbor_proj.shape[0], 1))), axis=1)
    h, status = cv2.findHomography(neighbor_proj, ref_proj)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, h, dsize=(width, height))

    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    zncc = np.zeros((src.shape[0], src.shape[1]))

    for h in range(src.shape[0]):
        for w in range(src.shape[1]):
            w1 = src[h, w].mean(axis=0)
            w2 = dst[h, w].mean(axis=0)
            sig1 = (((src[h, w] - w1) ** 2).mean(axis=0)) ** 0.5
            sig2 = (((dst[h, w] - w2) ** 2).mean(axis=0)) ** 0.5
            zncc[h, w] = (((src[h, w] - w1) * (dst[h, w] - w2)).sum(axis=0) / (sig1 * sig2 + EPS)).sum()
    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    x_map = (_u - K[0, 2]) * dep_map / K[0, 0]
    y_map = (_v - K[1, 2]) * dep_map / K[1, 1]
    xyz_cam = np.stack((x_map, y_map, dep_map), axis=2)
    """ END YOUR CODE
    """
    return xyz_cam

