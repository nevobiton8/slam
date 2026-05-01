import os
import cv2

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
