import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import (DOCS_PATH, read_images, detect_descriptors,
                   match_descriptors, reject_by_rectified_pattern,
                   read_cameras, triangulate_points)


def part2_1(kp_left, kp_right, matches):
    """Analyze the rectified stereo pattern of matches."""

    best_matches = [m[0] for m in matches]
    deviations = []
    for m in best_matches:
        y_left = kp_left[m.queryIdx].pt[1]
        y_right = kp_right[m.trainIdx].pt[1]
        deviations.append(abs(y_left - y_right))
    deviations = np.array(deviations)

    plt.figure(figsize=(10, 5))
    plt.hist(deviations, bins=50, edgecolor='black')
    plt.xlabel('deviation from rectified stereo pattern')
    plt.ylabel('Number of matches')
    plt.title('Part 2.1: Histogram of y-coordinate deviations')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, '2_1_deviation_histogram.png'), dpi=150)
    plt.show()

    pct_deviate = np.sum(deviations > 2) / len(deviations) * 100
    print(f"  Percentage of matches deviating by more than 2 pixels: {pct_deviate:.2f}%")

    return best_matches


def part2_2(img_left, img_right, kp_left, kp_right, best_matches):
    """Reject matches using the rectified stereo pattern."""
    inliers, outliers = reject_by_rectified_pattern(kp_left, kp_right, best_matches)

    print(f"\nPart 2.2:")
    print(f"  Total matches: {len(best_matches)}")
    print(f"  Inliers (accepted): {len(inliers)}")
    print(f"  Outliers (rejected): {len(outliers)}")

    # Plot matches as dots on both images
    # Draw outliers (cyan) first, then inliers (orange) on top
    img_height = img_left.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.imshow(img_left, cmap='gray')
    ax2.imshow(img_right, cmap='gray')

    # Outliers in cyan
    for m in outliers:
        pt_l = kp_left[m.queryIdx].pt
        pt_r = kp_right[m.trainIdx].pt
        ax1.plot(pt_l[0], pt_l[1], '.', color='cyan', markersize=3)
        ax2.plot(pt_r[0], pt_r[1], '.', color='cyan', markersize=3)

    # Inliers in orange (drawn second so they're on top)
    for m in inliers:
        pt_l = kp_left[m.queryIdx].pt
        pt_r = kp_right[m.trainIdx].pt
        ax1.plot(pt_l[0], pt_l[1], '.', color='orange', markersize=3)
        ax2.plot(pt_r[0], pt_r[1], '.', color='orange', markersize=3)

    ax1.set_title('Left image: inliers (orange), outliers (cyan)')
    ax1.axis('off')
    ax2.set_title('Right image: inliers (orange), outliers (cyan)')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, '2_2_rectified_rejection.png'), dpi=150)
    plt.show()

    # Under uniform distribution assumption, an erroneous match's right-image
    # y-coordinate is uniformly distributed over [0, img_height).
    # Probability of accepting an erroneous match = [y_left - 2, y_left + 2] / img_height.
    acceptance_band = 5
    prob_accept_error = acceptance_band / img_height
    prob_reject_error = 1 - prob_accept_error
    print(f"\n  Theoretical analysis (uniform distribution assumption):")
    print(f"    Image height: {img_height}px, acceptance band: {acceptance_band}px")
    print(f"    Probability of rejecting a random erroneous match: "
          f"{prob_reject_error:.4f} ({prob_reject_error*100:.2f}%)")
    num_rejected = len(outliers)
    # Estimated total erroneous matches = outliers / prob_reject_error
    est_total_erroneous = num_rejected / prob_reject_error
    est_wrongly_accepted = est_total_erroneous * prob_accept_error
    print(f"    Estimated erroneous matches wrongly accepted: "
          f"{est_wrongly_accepted:.1f}")

    return inliers, outliers


def part2_3(kp_left, kp_right, inlier_matches):
    """Triangulate 3D points using both custom and OpenCV methods."""
    k, m1, m2 = read_cameras()
    P1 = k @ m1
    P2 = k @ m2

    # Extract pixel coordinates
    pts_left = np.array([kp_left[m.queryIdx].pt for m in inlier_matches])
    pts_right = np.array([kp_right[m.trainIdx].pt for m in inlier_matches])

    # Custom triangulation
    points_custom = triangulate_points(P1, P2, pts_left, pts_right)

    # Plot custom triangulation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_custom[:, 0], points_custom[:, 1], points_custom[:, 2],
               s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Part 2.3: Custom linear least-squares triangulation')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, '2_3_triangulation_custom.png'), dpi=150)
    plt.show()

    # OpenCV triangulation
    pts_left_t = pts_left.T.astype(np.float64)   # 2xN
    pts_right_t = pts_right.T.astype(np.float64)  # 2xN
    points_cv_homo = cv2.triangulatePoints(P1, P2, pts_left_t, pts_right_t)
    points_cv = (points_cv_homo[:3] / points_cv_homo[3]).T  # Nx3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_cv[:, 0], points_cv[:, 1], points_cv[:, 2],
               s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Part 2.3: OpenCV triangulation')
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, '2_3_triangulation_opencv.png'), dpi=150)
    plt.show()

    # Compare results
    distances = np.linalg.norm(points_custom - points_cv, axis=1)
    median_dist = np.median(distances)
    print(f"\nPart 2.3:")
    print(f"  Number of triangulated points: {len(points_custom)}")
    print(f"  Median distance between custom and OpenCV 3D points: {median_dist:.6f}")

    return points_custom, k, m1, m2


def part2_4(k, m1, m2):
    """Run matching and triangulation on several stereo pairs."""
    P1 = k @ m1
    P2 = k @ m2
    frame_indices = [0, 50, 100, 200, 500]

    print(f"\nPart 2.4: Processing frames {frame_indices}")

    for idx in frame_indices:
        img_left, img_right = read_images(idx)
        kp_left, des_left = detect_descriptors(img_left)
        kp_right, des_right = detect_descriptors(img_right)
        matches = match_descriptors(des_left, des_right)
        best_matches = [m[0] for m in matches]
        inliers, outliers = reject_by_rectified_pattern(kp_left, kp_right, best_matches)

        pts_left = np.array([kp_left[m.queryIdx].pt for m in inliers])
        pts_right = np.array([kp_right[m.trainIdx].pt for m in inliers])
        points_3d = triangulate_points(P1, P2, pts_left, pts_right)

        # Identify erroneous points: extreme depth or negative Z
        median_z = np.median(points_3d[:, 2])
        erroneous = (points_3d[:, 2] < 0) | (points_3d[:, 2] > 10 * median_z)
        num_erroneous = np.sum(erroneous)

        print(f"\n  Frame {idx}:")
        print(f"    Matches: {len(best_matches)}, Inliers: {len(inliers)}, "
              f"Outliers: {len(outliers)}")
        print(f"    3D points: {len(points_3d)}, Erroneous (extreme depth): "
              f"{num_erroneous}")
        print(f"    Z range: [{points_3d[:, 2].min():.1f}, "
              f"{points_3d[:, 2].max():.1f}], median Z: {median_z:.1f}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        good_mask = ~erroneous
        ax.scatter(points_3d[good_mask, 0], points_3d[good_mask, 1],
                   points_3d[good_mask, 2], s=1, alpha=0.5, label='good')
        if num_erroneous > 0:
            ax.scatter(points_3d[erroneous, 0], points_3d[erroneous, 1],
                       points_3d[erroneous, 2], s=5, c='red', alpha=0.8,
                       label='erroneous')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {idx}: 3D point cloud')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DOCS_PATH, f'2_4_frame_{idx}.png'), dpi=150)
        plt.show()


def main():
    os.makedirs(DOCS_PATH, exist_ok=True)

    # Load first stereo pair and match (without significance test, as specified)
    img_left, img_right = read_images(0)
    kp_left, des_left = detect_descriptors(img_left)
    kp_right, des_right = detect_descriptors(img_right)
    matches = match_descriptors(des_left, des_right)

    best_matches = part2_1(kp_left, kp_right, matches)
    inliers, outliers = part2_2(img_left, img_right, kp_left, kp_right, best_matches)
    points_3d, k, m1, m2 = part2_3(kp_left, kp_right, inliers)
    part2_4(k, m1, m2)


if __name__ == '__main__':
    main()
