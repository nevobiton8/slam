import os
import cv2
import matplotlib.pyplot as plt
import random

from utils import (DOCS_PATH, read_images, detect_descriptors,
                   match_descriptors, apply_ratio_test)


def part1_1():
    """Detect and extract at least 500 keypoints on the first stereo pair.

    We use SIFT over alternatives (ORB, AKAZE) because:
    - SIFT produces float descriptors, which work well with L2 distance and
      Lowe's ratio test (the distance ratio is more meaningful in continuous space).
    - SIFT reliably detects well over 500 keypoints on KITTI images without
      parameter tuning.
    - ORB is faster but capped at a fixed number of keypoints by default, and its
      binary descriptors require Hamming distance which gives less granular ratio
      test results.
    """
    img_left, img_right = read_images(0)

    kp_left, des_left = detect_descriptors(img_left)
    kp_right, des_right = detect_descriptors(img_right)

    print(f"Part 1.1: Detected {len(kp_left)} keypoints in left image, "
          f"{len(kp_right)} keypoints in right image")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    img_left_kp = cv2.drawKeypoints(img_left, kp_left, None,
                                     color=(0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_right_kp = cv2.drawKeypoints(img_right, kp_right, None,
                                      color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ax1.imshow(img_left_kp)
    ax1.set_title(f'Left image - {len(kp_left)} keypoints')
    ax1.axis('off')

    ax2.imshow(img_right_kp)
    ax2.set_title(f'Right image - {len(kp_right)} keypoints')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, '1_1_keypoints.png'), dpi=150)
    plt.show()

    return img_left, img_right, kp_left, des_left, kp_right, des_right


def part1_2(des_left):
    """Print the descriptors of the two first features."""
    print("\nPart 1.2: Descriptors of the first two features (left image):")
    print(f"Feature 0 descriptor:\n{des_left[0]}")
    print(f"\nFeature 1 descriptor:\n{des_left[1]}")


def part1_3(img_left, img_right, kp_left, des_left, kp_right, des_right):
    """Match descriptors and present 20 random matches.

    We use BFMatcher (brute-force) with L2 norm because:
    - L2 is the natural distance metric for SIFT's float descriptors.
    - BFMatcher is exhaustive and guarantees finding the true nearest neighbor,
      acceptable here since the dataset is small.
    - k=2 returns the two closest matches per descriptor, which is required for
      Lowe's ratio test in part 1.4.
    """
    matches = match_descriptors(des_left, des_right)

    # Take only the best match from each pair for display
    best_matches = [m[0] for m in matches]

    random.seed(42)
    random_indices = random.sample(range(len(best_matches)), 20)
    random_matches = [best_matches[i] for i in random_indices]

    img_matches = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                                  random_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(16, 5))
    plt.imshow(img_matches)
    plt.title('Part 1.3: 20 Random Matches (before significance test)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, '1_3_matches.png'), dpi=150)
    plt.show()

    print(f"\nPart 1.3: Total matches found: {len(best_matches)}")

    return matches


def part1_4(img_left, img_right, kp_left, kp_right, matches):
    """Apply significance test (Lowe's ratio test) and report results.

    Lowe's ratio test rejects ambiguous matches where the best and second-best
    matches have similar distances — indicating the descriptor is not distinctive
    enough to identify a unique correspondence.

    To identify a correct match that was rejected, we exploit the epipolar constraint:
    in a rectified stereo pair, true correspondences must lie on the same horizontal
    scanline (same y-coordinate). A rejected match with a small y-difference (<2px)
    is therefore likely correct despite failing the ratio test.
    """
    ratio = 0.75
    good_matches, rejected_matches = apply_ratio_test(matches, ratio)

    num_discarded = len(rejected_matches)
    print(f"\nPart 1.4:")
    print(f"  Ratio value used: {ratio}")
    print(f"  Matches that passed: {len(good_matches)}")
    print(f"  Matches discarded: {num_discarded}")

    # Draw 20 random good matches
    random.seed(42)
    random_good = random.sample(good_matches, min(20, len(good_matches)))

    img_good = cv2.drawMatches(img_left, kp_left, img_right, kp_right,
                               random_good, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(16, 5))
    plt.imshow(img_good)
    plt.title(f'Part 1.4: 20 Random Matches after ratio test (ratio={ratio})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, '1_4_matches_after_ratio.png'), dpi=150)
    plt.show()

    # Find a correct match that was rejected by the significance test.
    # In a rectified stereo pair, correct matches should have similar y-coordinates.
    y_threshold = 2.0  # pixels
    correct_rejected = None
    for m in rejected_matches:
        pt_left = kp_left[m.queryIdx].pt
        pt_right = kp_right[m.trainIdx].pt
        if abs(pt_left[1] - pt_right[1]) < y_threshold:
            correct_rejected = m
            break

    # If not found, try with a stricter ratio to get more rejections
    if correct_rejected is None:
        for stricter_ratio in [0.6, 0.5, 0.4]:
            for m_pair in matches:
                m, n = m_pair
                if m.distance >= stricter_ratio * n.distance:
                    pt_left = kp_left[m.queryIdx].pt
                    pt_right = kp_right[m.trainIdx].pt
                    if abs(pt_left[1] - pt_right[1]) < y_threshold:
                        correct_rejected = m
                        print(f"  (Used stricter ratio {stricter_ratio} to find a "
                              f"correct rejected match)")
                        break
            if correct_rejected is not None:
                break

    if correct_rejected is not None:
        pt_left = kp_left[correct_rejected.queryIdx].pt
        pt_right = kp_right[correct_rejected.trainIdx].pt
        print(f"  Correct rejected match: left {pt_left} -> right {pt_right} "
              f"(y-diff: {abs(pt_left[1] - pt_right[1]):.2f}px)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        ax1.imshow(img_left, cmap='gray')
        ax1.plot(pt_left[0], pt_left[1], 'ro', markersize=10)
        ax1.set_title('Left image - rejected correct match')
        ax1.axis('off')

        ax2.imshow(img_right, cmap='gray')
        ax2.plot(pt_right[0], pt_right[1], 'ro', markersize=10)
        ax2.set_title('Right image - rejected correct match')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(DOCS_PATH, '1_4_rejected_correct_match.png'), dpi=150)
        plt.show()
    else:
        print("  Could not find a correct match that was rejected.")


def main():
    os.makedirs(DOCS_PATH, exist_ok=True)

    img_left, img_right, kp_left, des_left, kp_right, des_right = part1_1()
    part1_2(des_left)
    matches = part1_3(img_left, img_right, kp_left, des_left, kp_right, des_right)
    part1_4(img_left, img_right, kp_left, kp_right, matches)


if __name__ == '__main__':
    main()
