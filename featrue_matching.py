import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def load_images_and_pcd(img1_path, img2_path, pcd1_path, pcd2_path):
    """Load grayscale and color images, and point cloud data."""
    img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)
    pcd1 = o3d.io.read_point_cloud(pcd1_path)
    pcd2 = o3d.io.read_point_cloud(pcd2_path)
    return img1_gray, img2_gray, img1_color, img2_color, pcd1, pcd2

def load_camera_parameters(intrinsic_path, extrinsic_path, binning=2):
    """Load intrinsic and extrinsic camera parameters."""
    intrinsic_param = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
    intrinsic = np.array([[intrinsic_param[0]/binning, intrinsic_param[1], intrinsic_param[2]/binning],
                          [0.0, intrinsic_param[3]/binning, intrinsic_param[4]/binning],
                          [0.0, 0.0, 1.0]])
    distortion = np.array(intrinsic_param[5:])
    extrinsic = np.loadtxt(extrinsic_path, delimiter=',')
    extrinsic.shape = (4, 4)
    extrinsic_inv = np.linalg.inv(extrinsic)
    return intrinsic, distortion, extrinsic, extrinsic_inv

def visualize_roi(img1_color, img2_color, roi):
    """Visualize ROI on input images."""
    img1_roi_vis = img1_color.copy()
    img2_roi_vis = img2_color.copy()
    cv2.rectangle(img1_roi_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
    cv2.rectangle(img2_roi_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
    cv2.imshow('Image1 with ROI', img1_roi_vis)
    cv2.imshow('Image2 with ROI', img2_roi_vis)

def extract_and_match_features(img1_gray, img2_gray, roi):
    """Extract SIFT features from ROI and perform FLANN-based matching."""
    img1_roi = img1_gray[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    img2_roi = img2_gray[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_roi, None)
    kp2, des2 = sift.detectAndCompute(img2_roi, None)
    kp1 = [cv2.KeyPoint(k.pt[0] + roi[0], k.pt[1] + roi[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp1]
    kp2 = [cv2.KeyPoint(k.pt[0] + roi[0], k.pt[1] + roi[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp2]
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    reconnection_attempts = 3
    for attempt in range(reconnection_attempts):
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == reconnection_attempts - 1:
                raise Exception("Failed to perform FLANN matching after multiple attempts")
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return kp1, kp2, des1, des2, good

def compute_homography_and_visualize(img1_color, img2_color, kp1, kp2, good):
    """Compute homography, warp image, and visualize results."""
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                           matchesMask=matches_mask, flags=2)
        img_match = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good, None, **draw_params)
        cv2.imshow('ROI Feature Matching', img_match)
        img1_warped = cv2.warpPerspective(img1_color, M, (img2_color.shape[1], img2_color.shape[0]))
        cv2.imshow('Warped Image1 (Homography)', img1_warped)
        blend = cv2.addWeighted(img1_warped, 0.4, img2_color, 0.6, 0)
        cv2.imshow('Blended Warped vs Image2', blend)
        return src_pts, dst_pts, M
    else:
        print("매칭점 부족")
        return None, None, None

def project_points_to_image(pcd_points, intrinsic, distortion, extrinsic):
    """Project 3D point cloud to 2D image plane."""
    pcd_homo = np.hstack([pcd_points, np.ones((pcd_points.shape[0], 1))])
    pcd_cam = (extrinsic @ pcd_homo.T).T[:, :3]
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    proj, _ = cv2.projectPoints(pcd_cam.astype(np.float32), rvec, tvec, intrinsic, distortion)
    uv = proj.squeeze()
    return uv, pcd_cam

def filter_valid_points(uv, pcd_points, pcd_cam, img_shape):
    """Filter valid projected points within image boundaries and in front of camera."""
    valid = (pcd_cam[:,2] > 0) \
        & (uv[:,0] >= 0) & (uv[:,0] < img_shape[1]) \
        & (uv[:,1] >= 0) & (uv[:,1] < img_shape[0])
    uv_valid = uv[valid]
    pcd_valid = pcd_points[valid]
    return uv_valid, pcd_valid

def match_2d_to_3d(src_pts, uv_valid, pcd_valid):
    """Match 2D feature points to 3D point cloud using KDTree."""
    src_pts_2d = src_pts.reshape(-1, 2)
    tree2d = cKDTree(uv_valid)
    dist, idx = tree2d.query(src_pts_2d, distance_upper_bound=5)
    matched_3d = [pcd_valid[idx[i]] for i, d in enumerate(dist) if d != np.inf]
    return np.array(matched_3d), dist, idx

def visualize_3d_points(pcd1, correspond_3d_points, extrinsic_inv):
    """Visualize matched 3D points and coordinate frames."""
    pcd_matched = o3d.geometry.PointCloud()
    pcd_matched.points = o3d.utility.Vector3dVector(correspond_3d_points)
    pcd_matched.paint_uniform_color([1.0, 0.0, 0.0])
    spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.15).translate(pt).paint_uniform_color([1.0, 0.0, 0.0])
               for pt in correspond_3d_points]
    pcd1.paint_uniform_color([0.7, 0.7, 0.7])
    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3).transform(extrinsic_inv)
    o3d.visualization.draw_geometries([pcd1, pcd_matched, lidar_frame, cam_frame] + spheres,
                                      point_show_normal=False, width=1280, height=720,
                                      window_name='Matched Points Visualization')

def visualize_2d_matches(img1_color, src_pts, uv_valid, dist, roi):
    """Visualize 2D-3D correspondences on the image."""
    img_vis = img1_color.copy()
    src_pts_2d = src_pts.reshape(-1, 2)
    valid_mask = (dist != np.inf)
    invalid_mask = (dist == np.inf)
    matched_2d = src_pts_2d[valid_mask]
    unmatched_2d = src_pts_2d[invalid_mask]
    for u, v in matched_2d:
        u, v = int(round(u)), int(round(v))
        cv2.circle(img_vis, (u, v), 8, (0, 0, 255), 2)
        cv2.drawMarker(img_vis, (u, v), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    for u, v in unmatched_2d:
        u, v = int(round(u)), int(round(v))
        cv2.circle(img_vis, (u, v), 8, (255, 0, 0), 2)
        cv2.drawMarker(img_vis, (u, v), (255,255,0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    for uu, vv in uv_valid.astype(int):
        cv2.circle(img_vis, (uu, vv), 1, (255,255,255), -1)
    cv2.rectangle(img_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
    cv2.imshow("Image with Matched 3D Correspondence", img_vis)

def rigid_transform_3D(A, B):
    """
    A, B: Nx3 numpy arrays (A: source, B: target)
    Returns R, t such that: B ≈ R @ A.T + t
    """
    assert A.shape == B.shape

    # 1. 중심화
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # 2. 공분산
    H = AA.T @ BB

    # 3. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection(좌우반전) 보정
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


def main():
    # File paths
    img1_path = '/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/camera1_image_color_compressed/2025-05-16_103546536693.jpg'
    img2_path = '/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-17/cam1/2025-05-17_13-03-50.png'
    pcd1_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/ouster1_points/2025-05-16_103546536693.pcd"
    pcd2_path = "0424-0515/2025-05-17/os1/2025-05-17_13-03-50.pcd"
    intrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/intrinsic1.csv"
    extrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/iou_optimized_transform1.txt"
    roi = (500, 50, 500, 130)  # (x, y, w, h)

    # Load data
    img1_gray, img2_gray, img1_color, img2_color, pcd1, pcd2 = load_images_and_pcd(img1_path, img2_path, pcd1_path, pcd2_path)
    intrinsic, distortion, extrinsic, extrinsic_inv = load_camera_parameters(intrinsic_path, extrinsic_path, binning=2)
    print("Camera Matrix:\n", intrinsic)
    print("Distortion Coefficients:\n", distortion)
    print("Camera-Lidar Extrinsic:\n", extrinsic)
    print("Camera-Lidar Extrinsic Inverse:\n", extrinsic_inv)

    # Visualize ROI
    visualize_roi(img1_color, img2_color, roi)

    # Feature extraction and matching
    kp1, kp2, des1, des2, good = extract_and_match_features(img1_gray, img2_gray, roi)

    # Homography computation and visualization
    src_pts, dst_pts, M = compute_homography_and_visualize(img1_color, img2_color, kp1, kp2, good)
    if src_pts is None:
        return

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Point cloud processing - Past data    
    pcd_points = np.asarray(pcd1.points)
    uv, pcd_cam = project_points_to_image(pcd_points, intrinsic, distortion, extrinsic)
    uv_valid, pcd_valid = filter_valid_points(uv, pcd_points, pcd_cam, img1_color.shape)
    correspond_3d_points, dist, idx = match_2d_to_3d(src_pts, uv_valid, pcd_valid)

    # Visualize 3D points
    visualize_3d_points(pcd1, correspond_3d_points, extrinsic_inv)

    # Visualize 2D matches
    visualize_2d_matches(img1_color, src_pts, uv_valid, dist, roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Point cloud processing - Current data
    pcd_points_curr = np.asarray(pcd2.points)
    uv_curr, pcd_cam_curr = project_points_to_image(pcd_points_curr, intrinsic, distortion, extrinsic)
    uv_valid_curr, pcd_valid_curr = filter_valid_points(uv_curr, pcd_points_curr, pcd_cam_curr, img2_color.shape)
    correspond_3d_points_curr, dist_curr, idx_curr = match_2d_to_3d(dst_pts, uv_valid_curr, pcd_valid_curr)
    # Visualize 3D points
    visualize_3d_points(pcd2, correspond_3d_points_curr, extrinsic_inv)
    # Visualize 2D matches
    visualize_2d_matches(img2_color, dst_pts, uv_valid_curr, dist_curr, roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 과거 correspondence(매칭 성공한 것만)  
    valid_mask_past = (dist != np.inf)
    # 현재 correspondence(매칭 성공한 것만)  
    valid_mask_now = (dist_curr != np.inf)

    # 매칭된 전체 인덱스는 FLANN good match 개수 기준임
    assert len(valid_mask_past) == len(valid_mask_now)  # 둘 다 good match 개수와 동일

    # 1:1로 대응되는 인덱스만 남기기 (둘 다 성공한 것만)
    joint_mask = valid_mask_past & valid_mask_now

# 3D correspondence는 2D와 직접적으로 1:1 대응(매칭이 성공/실패)하는지 "dist/idx" 기준으로 생성

    A_list = []
    B_list = []
    N = len(src_pts)  # good match 개수

    for i in range(N):
        # dist[i]: 과거 PCD, dist_curr[i]: 현재 PCD
        if (dist[i] != np.inf) and (dist_curr[i] != np.inf):
            # pcd_valid와 idx로 바로 접근 (순서를 보장)
            A_list.append(pcd_valid_curr[idx_curr[i]])  # 현재 3D점 (dst_pts→pcd_valid_curr)
            B_list.append(pcd_valid[idx[i]])            # 과거 3D점 (src_pts→pcd_valid)
    # Numpy array로 변환
    A = np.array(A_list)
    B = np.array(B_list)
    print("최종 correspondence 쌍 개수:", len(A))



    R, t = rigid_transform_3D(A, B)
    # 변환 행렬로 표현
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    print("Rigid 변환 행렬:\n", T)

    # 예시: 현재 PCD의 모든 점을 과거 좌표계로 이동
    pcd_now_aligned = (R @ pcd_points_curr.T).T + t  # (N,3)

    o3d_pcd_now_aligned = o3d.geometry.PointCloud()
    o3d_pcd_now_aligned.points = o3d.utility.Vector3dVector(pcd_now_aligned)
    o3d_pcd_now_aligned.paint_uniform_color([0.0, 1.0, 0.0])  # Green

    # Visualize aligned point cloud
    o3d_pcd1 = pcd1
    o3d_pcd1.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    o3d_pcd2 = pcd2
    o3d_pcd2.paint_uniform_color([0.0, 0.0, 1.0])  # Gray
    o3d.visualization.draw_geometries([o3d_pcd_now_aligned, o3d_pcd2],
                                      point_show_normal=False, width=1280, height=720,
                                      window_name='Aligned Point Clouds vs Current PCD')
    
    o3d_pcd1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d_pcd_now_aligned_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d_pcd_now_aligned_frame.transform(T)
    o3d.visualization.draw_geometries([o3d_pcd1, o3d_pcd_now_aligned, o3d_pcd1_frame, o3d_pcd_now_aligned_frame],
                                    point_show_normal=False, width=1280, height=720,
                                    window_name='Aligned Point Clouds vs Past PCD')
    cv2.waitKey(0)

if __name__ == "__main__":
    main()