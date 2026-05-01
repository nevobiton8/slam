import os
import cv2
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'sequences', '00')
DOCS_PATH = os.path.join(os.path.dirname(__file__), '..', 'docs')


def read_images(idx):
    """Read a stereo pair (left, right) by frame index."""
    img_name = '{:06d}.png'.format(idx)
    img_left = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name), 0)
    img_right = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name), 0)
    return img_left, img_right


def detect_descriptors(img):
    """Detect SIFT keypoints and compute descriptors for a single image."""
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


def match_descriptors(des1, des2):
    """Match two descriptor sets using BFMatcher with L2 norm (kNN, k=2)."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    return matches


def apply_ratio_test(matches, ratio=0.75):
    """Apply Lowe's ratio test. Returns (good_matches, rejected_matches)."""
    good = []
    rejected = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
        else:
            rejected.append(m)
    return good, rejected


def reject_by_rectified_pattern(kp_left, kp_right, matches, y_threshold=2.0):
    """Reject matches where y-coordinates deviate beyond threshold in rectified stereo.
    Returns (inliers, outliers)."""
    inliers = []
    outliers = []
    for m in matches:
        y_left = kp_left[m.queryIdx].pt[1]
        y_right = kp_right[m.trainIdx].pt[1]
        if abs(y_left - y_right) <= y_threshold:
            inliers.append(m)
        else:
            outliers.append(m)
    return inliers, outliers


def read_cameras():
    """Read camera projection matrices from calib.txt.
    Returns (K, m1, m2) where K is the intrinsic matrix and m1, m2 are
    the extrinsic matrices for the left and right cameras."""
    with open(os.path.join(DATA_PATH, 'calib.txt')) as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    l2 = [float(i) for i in l2]
    p0 = np.array(l1).reshape(3, 4)
    p1 = np.array(l2).reshape(3, 4)
    k = p0[:, :3]
    m1 = np.linalg.inv(k) @ p0
    m2 = np.linalg.inv(k) @ p1
    return k, m1, m2


def triangulate_points(p1, p2, pts1, pts2):
    """Linear least-squares triangulation using SVD.
    p1, p2: 3x4 projection matrices.
    pts1, pts2: Nx2 arrays of corresponding pixel coordinates.
    Returns Nx3 array of 3D points."""
    n = pts1.shape[0]
    points_3d = np.zeros((n, 3))
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.array([# This is the simplified linear least squares equations in a matrix
            x1 * p1[2] - p1[0],
            y1 * p1[2] - p1[1],
            x2 * p2[2] - p2[0],
            y2 * p2[2] - p2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]# Smallest singular value vector, which solves the Linear Programming optimization problem
        points_3d[i] = X[:3] / X[3]# conversion from 4D homo to 3D
    return points_3d
