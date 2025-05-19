import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from copy import deepcopy
import os
from datetime import datetime
from scipy.optimize import least_squares
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


# 출력 디렉토리 설정
OUTPUT_DIR = "./outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_image(img, name_prefix):
    """Save OpenCV image with timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{name_prefix}_{timestamp}.png")
    cv2.imwrite(filename, img)
    print(f"Saved image: {filename}")

def save_open3d_screenshot(vis, name_prefix):
    """Capture and save Open3D visualization screenshot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{name_prefix}_{timestamp}.png")
    vis.capture_screen_image(filename)
    print(f"Saved Open3D screenshot: {filename}")

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
    """Visualize ROI on input images and save."""
    img1_roi_vis = img1_color.copy()
    img2_roi_vis = img2_color.copy()
    cv2.rectangle(img1_roi_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
    cv2.rectangle(img2_roi_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
    cv2.imshow('Image1 with ROI', img1_roi_vis)
    cv2.imshow('Image2 with ROI', img2_roi_vis)
    save_image(img1_roi_vis, "roi_image1")
    save_image(img2_roi_vis, "roi_image2")

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
    """Compute homography (current → past), visualize, and save results."""
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # 과거
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # 현재
        # 현재 → 과거 호모그래피
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                           matchesMask=matches_mask, flags=2)
        img_match = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good, None, **draw_params)
        cv2.imshow('ROI Feature Matching', img_match)
        save_image(img_match, "feature_matching")
        # 현재 이미지를 과거 좌표계로 워핑
        img2_warped = cv2.warpPerspective(img2_color, M, (img1_color.shape[1], img1_color.shape[0]))
        cv2.imshow('Warped Image2 (Homography)', img2_warped)
        save_image(img2_warped, "warped_image2")
        blend = cv2.addWeighted(img1_color, 0.6, img2_warped, 0.4, 0)
        cv2.imshow('Blended Image1 vs Warped Image2', blend)
        save_image(blend, "blended_warped")
        return src_pts, dst_pts, M
    else:
        print("매칭점 부족")
        return None, None, None

def get_corners(shape):
    """Return image corners in homogeneous coordinates."""
    h, w = shape[:2]
    return np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T

def overlay_union_canvas(img_past, img_current, H, roi, alpha=0.5):
    """Create alpha-blended canvas with borders and save (current → past)."""
    corners_past = get_corners(img_past.shape)
    corners_current = get_corners(img_current.shape)
    corners_current_warped = H @ corners_current  # H is current → past
    corners_current_warped = (corners_current_warped[:2] / corners_current_warped[2:]).T
    all_pts = np.vstack([corners_past[:2, :].T, corners_current_warped])
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    width = x_max - x_min
    height = y_max - y_min
    canvas_past = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_current = np.zeros_like(canvas_past)
    x_offset_past = -x_min
    y_offset_past = -y_min
    canvas_past[y_offset_past:y_offset_past + img_past.shape[0], x_offset_past:x_offset_past + img_past.shape[1]] = img_past
    offset_mat = np.array([[1, 0, x_offset_past], [0, 1, y_offset_past], [0, 0, 1]])
    H_offset = offset_mat @ H
    img_current_warped = cv2.warpPerspective(img_current, H_offset, (width, height))
    canvas_current[:,:,:] = img_current_warped
    blend = cv2.addWeighted(canvas_past, alpha, canvas_current, 1 - alpha, 0)
    cv2.rectangle(blend, (x_offset_past, y_offset_past),
                  (x_offset_past + img_past.shape[1] - 1, y_offset_past + img_past.shape[0] - 1), (0, 255, 0), 1)
    contour_pts = corners_current_warped + np.array([x_offset_past, y_offset_past])
    contour_pts = contour_pts.astype(int).reshape((-1, 1, 2))
    cv2.polylines(blend, [contour_pts], isClosed=True, color=(255, 0, 0), thickness=1)
    x, y, w, h = roi
    cv2.rectangle(blend, (x_offset_past + x, y_offset_past + y),
                  (x_offset_past + x + w, y_offset_past + y + h), (0, 0, 255), 1)
    cv2.imshow("Union Alpha Blend (addWeighted)", blend)
    save_image(blend, "union_alpha_blend")
    return blend

def decompose_homography(H, K):
    """Decompose homography to estimate relative pose (current → past)."""
    num, Rs, Ts, normals = cv2.decomposeHomographyMat(H, K)
    valid_poses = []
    for i in range(num):
        R = Rs[i]
        t = Ts[i]
        n = normals[i]
        if t[2] > 0:  # 카메라가 평면 앞에 있는지 확인
            valid_poses.append((R, t, n))
    return valid_poses

def evaluate_pose2(img1_color, roi, correspond_3d_points, correspond_3d_points_curr, src_pts, R, t, intrinsic, distortion, extrinsic, visualize=True):
    """Evaluate pose using reprojection error (current → past)."""
    errors = []
    src_pts_2d = src_pts.reshape(-1, 2)  # 과거 2D 점
    T = np.eye(4)
    T[:3, :3] = R  # R_curr→past
    T[:3, 3] = t.flatten()  # t_curr→past
    for pt_3d_curr, pt_2d_past in zip(correspond_3d_points_curr, src_pts_2d):
        pt_3d_homo = np.append(pt_3d_curr, 1)
        pt_3d_past = (T @ pt_3d_homo)[:3]  # 현재 3D 점을 과거 좌표계로 변환
        pt_3d_cam = (extrinsic @ np.append(pt_3d_past, 1))[:3]
        pt_2d, _ = cv2.projectPoints(pt_3d_cam.reshape(1, -1), np.zeros(3), np.zeros(3), intrinsic, distortion)
        pt_2d = pt_2d.squeeze()
        error = np.linalg.norm(pt_2d - pt_2d_past)
        if error < 50:
            errors.append(error)
    mean_error = np.mean(errors) if errors else float('inf')
    valid_ratio = len(errors) / len(src_pts_2d) if len(src_pts_2d) > 0 else 0
    if visualize and errors:
        img_vis = img1_color.copy()
        for pt_3d_curr, pt_2d_past in zip(correspond_3d_points_curr, src_pts_2d):
            pt_3d_homo = np.append(pt_3d_curr, 1)
            pt_3d_past = (T @ pt_3d_homo)[:3]
            pt_3d_cam = (extrinsic @ np.append(pt_3d_past, 1))[:3]
            pt_2d, _ = cv2.projectPoints(pt_3d_cam.reshape(1, -1), np.zeros(3), np.zeros(3), intrinsic, distortion)
            pt_2d = pt_2d.squeeze().astype(int)
            cv2.circle(img_vis, tuple(pt_2d), 5, (0, 255, 0), 2)  # 투영 점 (녹색)
            cv2.circle(img_vis, tuple(pt_2d_past.astype(int)), 5, (0, 0, 255), 2)  # 원래 점 (빨강)
            cv2.line(img_vis, tuple(pt_2d), tuple(pt_2d_past.astype(int)), (255, 0, 0), 1)  # 오차 선 (파랑)
        cv2.rectangle(img_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
        cv2.imshow(f"Reprojection Error (Mean: {mean_error:.2f} pixels)", img_vis)
        save_image(img_vis, f"reprojection_error_{mean_error:.2f}")
    return mean_error, valid_ratio

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
    valid = (pcd_cam[:,2] > 0) & (uv[:,0] >= 0) & (uv[:,0] < img_shape[1]) & (uv[:,1] >= 0) & (uv[:,1] < img_shape[0])
    uv_valid = uv[valid]
    pcd_valid = pcd_points[valid]
    return uv_valid, pcd_valid

def match_2d_to_3d(src_pts, uv_valid, pcd_valid):
    """Match 2D points to 3D points using k-d tree."""
    src_pts_2d = src_pts.reshape(-1, 2)
    tree2d = cKDTree(uv_valid)
    dist, idx = tree2d.query(src_pts_2d, distance_upper_bound=3)
    valid_mask = dist != np.inf
    matched_3d = pcd_valid[idx[valid_mask]]
    valid_indices = np.where(valid_mask)[0]
    return matched_3d, dist, idx, valid_indices

def visualize_3d_points(pcd, correspond_3d_points, extrinsic_inv, window_name):
    """Visualize matched 3D points and coordinate frames, and save."""
    pcd_matched = o3d.geometry.PointCloud()
    pcd_matched.points = o3d.utility.Vector3dVector(correspond_3d_points)
    pcd_matched.paint_uniform_color([1.0, 0.0, 0.0])
    spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.15).translate(pt).paint_uniform_color([1.0, 0.0, 0.0])
               for pt in correspond_3d_points]
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3).transform(extrinsic_inv)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    for geom in [pcd, pcd_matched, lidar_frame, cam_frame] + spheres:
        vis.add_geometry(geom)
    vis.run()
    save_open3d_screenshot(vis, f"3d_points_{window_name.lower().replace(' ', '_')}")
    vis.destroy_window()

def visualize_2d_matches(img_color, src_pts, uv_valid, dist, roi, window_name):
    """Visualize 2D-3D correspondences on the image and save."""
    img_vis = img_color.copy()
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
    cv2.imshow(window_name, img_vis)
    save_image(img_vis, f"2d_matches_{window_name.lower().replace(' ', '_')}")

def find_relative_pose(img, pcd1, pcd2, roi, extrinsic, intrinsic, distortion, initial=None):
    # pcd1, pcd2: (N, 3) array, 반드시 1:1 대응!
    assert pcd1.shape == pcd2.shape

    # 1. extrinsic으로 라이다 좌표를 카메라 좌표계로 변환
    def transform_points(pts, T):
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_cam = (T @ pts_homo.T).T[:, :3]
        return pts_cam

    pcd1_cam = transform_points(pcd1, extrinsic)
    pcd2_cam = transform_points(pcd2, extrinsic)

    if initial is not None:
        pcd2_cam = transform_points(pcd2_cam, initial)

    # 2. cost function 정의: (R, t)를 적용한 pcd1이 pcd2와 이미지 투영시 최소 거리
    def reprojection_error(x):
        # x[:3] = rotation vector (axis-angle), x[3:] = translation
        rotvec = x[:3]
        tvec = x[3:]
        rot = R.from_rotvec(rotvec).as_matrix()
        pcd2_transformed = (rot @ pcd2_cam.T).T + tvec

        proj1, _ = cv2.projectPoints(pcd1_cam.astype(np.float32), np.zeros(3), np.zeros(3), intrinsic, distortion)
        proj2, _ = cv2.projectPoints(pcd2_transformed.astype(np.float32), np.zeros(3), np.zeros(3), intrinsic, distortion)
        proj1 = proj1.squeeze()
        proj2 = proj2.squeeze()
        return (proj1 - proj2).ravel()  # 2N residual

    # 3. 초기값: identity (0, 0, 0, 0, 0, 0)
    x0 = np.zeros(6)
    result = least_squares(reprojection_error, x0, method='lm', verbose=2, xtol=1e-9, ftol=1e-9, gtol=1e-9)

    # 4. 최적 파라미터 추출 및 변환 행렬 생성
    rotvec_opt = result.x[:3]
    tvec_opt = result.x[3:]
    rot_opt = R.from_rotvec(rotvec_opt).as_matrix()
    T_opt = np.eye(4)
    T_opt[:3, :3] = rot_opt
    T_opt[:3, 3] = tvec_opt

    print("최적 Rotation (Rodrigues):", rotvec_opt)
    print("최적 Translation:", tvec_opt)
    print("최적 Transformation:\n", T_opt)
    print("최종 reprojection error:", np.linalg.norm(result.fun.reshape(-1,2), axis=1).mean(), "(평균 픽셀)")

    # 5. 시각화 (Projected points)
    projection_img = img.copy()
    pcd2_transformed = (rot_opt @ pcd2_cam.T).T + tvec_opt
    proj1, _ = cv2.projectPoints(pcd1_cam.astype(np.float32), np.zeros(3), np.zeros(3), intrinsic, distortion)
    proj2, _ = cv2.projectPoints(pcd2_transformed.astype(np.float32), np.zeros(3), np.zeros(3), intrinsic, distortion)
    proj1 = proj1.squeeze().astype(int)
    proj2 = proj2.squeeze().astype(int)
    for u,v in proj1:
        cv2.circle(projection_img, (u, v), 5, (0, 0, 255), 2)
    for u,v in proj2:
        cv2.circle(projection_img, (u, v), 5, (255, 0, 0), 2)
    cv2.rectangle(projection_img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
    cv2.imshow("Projected Points (Leastsq aligned)", projection_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return T_opt, rot_opt, tvec_opt

def main():
    # File paths
    img1_path = '0424-0515/2025-04-22/cam1/2025-04-22_17-19-59.png'
    img2_path = '/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-17/cam1/2025-05-17_13-03-50.png'
    pcd1_path = "0424-0515/2025-04-22/os1/2025-04-22_17-19-59.pcd"
    pcd2_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-17/os1/2025-05-17_13-03-50.pcd"
    intrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/intrinsic1.csv"
    extrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/iou_optimized_transform1.txt"
    roi = (80, 0, 910, 150)

    # Load data
    global img1_color
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

    # Homography computation and visualization (current → past)
    src_pts, dst_pts, M = compute_homography_and_visualize(img1_color, img2_color, kp1, kp2, good)
    if src_pts is None:
        return

    print("Homography Matrix (current → past):\n", M)
    overlay_union_canvas(img1_color, img2_color, M, roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Point cloud processing - Past data
    pcd_points = np.asarray(pcd1.points)
    uv, pcd_cam = project_points_to_image(pcd_points, intrinsic, distortion, extrinsic)
    uv_valid, pcd_valid = filter_valid_points(uv, pcd_points, pcd_cam, img1_color.shape)
    correspond_3d_points, dist, idx, valid_idx = match_2d_to_3d(src_pts, uv_valid, pcd_valid)
    correspond_3d_pcd = o3d.geometry.PointCloud()
    correspond_3d_pcd.points = o3d.utility.Vector3dVector(correspond_3d_points)
    visualize_3d_points(pcd1, correspond_3d_points, extrinsic_inv, "Past 3D Points")
    visualize_2d_matches(img1_color, src_pts, uv_valid, dist, roi, "Past 2D Matches")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Point cloud processing - Current data
    pcd_points_curr = np.asarray(pcd2.points)
    uv_curr, pcd_cam_curr = project_points_to_image(pcd_points_curr, intrinsic, distortion, extrinsic)
    uv_valid_curr, pcd_valid_curr = filter_valid_points(uv_curr, pcd_points_curr, pcd_cam_curr, img2_color.shape)
    correspond_3d_points_curr, dist_curr, idx_curr, valid_idx_curr = match_2d_to_3d(dst_pts, uv_valid_curr, pcd_valid_curr)
    correspond_3d_pcd_curr = o3d.geometry.PointCloud()
    correspond_3d_pcd_curr.points = o3d.utility.Vector3dVector(correspond_3d_points_curr)
    visualize_3d_points(pcd2, correspond_3d_points_curr, extrinsic_inv, "Current 3D Points")
    visualize_2d_matches(img2_color, dst_pts, uv_valid_curr, dist_curr, roi, "Current 2D Matches")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    if M is not None:
        # 호모그래피 분해 (current → past)
        valid_poses = decompose_homography(M, intrinsic)
        print(f"Found {len(valid_poses)} valid pose solutions")
        for i, (R, t, n) in enumerate(valid_poses):
            print(f"Pose {i+1}:")
            print("Rotation Matrix (R_curr→past):\n", R)
            print("Translation Vector (t_curr→past):\n", t)
            print("Plane Normal (n):\n", n)

        if valid_poses:
            # 일치하는 특징점-3D 점 쌍 필터링
            valid_mask_past = (dist != np.inf)
            valid_mask_now = (dist_curr != np.inf)
            assert len(valid_mask_past) == len(valid_mask_now)
            joint_mask = valid_mask_past & valid_mask_now
            matched_3d_curr = []
            matched_3d_past = []
            matched_src_pts = []
            N = len(src_pts)
            for i in range(N):
                if joint_mask[i]:
                    matched_3d_curr.append(pcd_valid_curr[idx_curr[i]])
                    matched_3d_past.append(pcd_valid[idx[i]])
                    matched_src_pts.append(src_pts[i])  # 과거 2D 점
            
            matched_3d_curr = np.array(matched_3d_curr)
            matched_3d_past = np.array(matched_3d_past)
            matched_src_pts = np.array(matched_src_pts)


            print("최종 correspondence 쌍 개수:", len(matched_3d_past), len(matched_3d_curr))

            if len(matched_3d_past) < 4:
                print("Insufficient matched 3D points for evaluation")
                return

            # 포즈 평가
            best_pose = None
            best_fitness = float('inf')
            for i, (R, t, n) in enumerate(valid_poses):
                mean_error, valid_ratio = evaluate_pose2(
                    img1_color, roi, matched_3d_past, matched_3d_curr, matched_src_pts, R, t,
                    intrinsic, distortion, extrinsic, visualize=True
                )
                print(f"Pose {i+1}: Mean Reprojection Error={mean_error:.2f} pixels, Valid Ratio={valid_ratio:.2%}")
                if valid_ratio > 0.5 and mean_error < best_fitness:
                    best_fitness = mean_error
                    best_pose = (R, t, n)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if best_pose:
                R, t, n = best_pose
                # 스케일 보정
                def estimate_plane_distance(pcd, extrinsic):
                    points = np.asarray(pcd.points)
                    pcd_cam = (extrinsic @ np.hstack([points, np.ones((points.shape[0], 1))]).T).T[:, :3]
                    return np.mean(pcd_cam[:, 2])
                d = estimate_plane_distance(pcd2, extrinsic)  # 현재 프레임 기준
                t = t * d
                print("Best Pose from Homography (Current→Past):")
                print("Rotation Matrix (R_curr→past):\n", R)
                print("Translation Vector (t_curr→past):\n", t)
                # print("Plane Normal (n):\n", n)
                relative_pose = np.eye(4)
                relative_pose[:3, :3] = R
                relative_pose[:3, 3] = t.flatten()
                relative_pose_inv = np.linalg.inv(relative_pose)
                # 좌표계 시각화

                T_opt, rot_opt, tvec_opt = find_relative_pose(img1_color, matched_3d_curr, matched_3d_past, 
                                                              roi, extrinsic, intrinsic, distortion, initial=relative_pose_inv)
                T_opt_inv = np.linalg.inv(T_opt)

                transform = T_opt

                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='Coordinate Frames', width=1280, height=720)
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).transform(extrinsic)
                cam_frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3).transform(transform)
                lidar_frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).transform(transform @ extrinsic)
                lidar_frame.paint_uniform_color([0.0, 0.0, 1.0])  # 현재 라이다 (파랑)
                lidar_frame_est.paint_uniform_color([0.0, 1.0, 0.0])  # 과거 라이다 (녹색)
                cam_frame.paint_uniform_color([0.0, 0.0, 1.0])  # 현재 카메라 (파랑)
                cam_frame_est.paint_uniform_color([0.0, 1.0, 0.0])  # 과거 카메라 (녹색)
                for geom in [cam_frame, cam_frame_est, lidar_frame, lidar_frame_est]:
                    vis.add_geometry(geom)
                vis.run()
                save_open3d_screenshot(vis, "coordinate_frames")
                vis.destroy_window()

                # 포인트 클라우드 시각화
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='Matched Points Visualization', width=1280, height=720)
                pcd1_temp = deepcopy(pcd1)
                pcd2_temp = deepcopy(pcd2)
                pcd2_temp.transform(T_opt_inv @extrinsic)  # 현재 프레임
                pcd1_temp = pcd1.transform(extrinsic)  # 과거 프레임
                pcd2_temp.paint_uniform_color([0.0, 0.0, 1.0])  # 현재 (파랑)
                pcd1_temp.paint_uniform_color([0.5, 0.5, 0.5])  # 과거 (녹색)
                for geom in [pcd1_temp, pcd2_temp, lidar_frame, lidar_frame_est, cam_frame, cam_frame_est]:
                    vis.add_geometry(geom)
                vis.run()
                save_open3d_screenshot(vis, "matched_points_final")
                vis.destroy_window()
                pcd2_temp = deepcopy(pcd2)
                pcd2_temp.transform(relative_pose)
                o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "matched_points_final.pcd"), pcd2_temp)

if __name__ == "__main__":
    main()