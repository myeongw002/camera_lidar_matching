import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from copy import deepcopy


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

def compute_homography_and_visualize_reverse(img1_color, img2_color, kp1, kp2, good, roi):
    """
    과거 이미지 좌표계로 현재 이미지를 전체 warp 후 blend.
    각 이미지의 경계 윤곽선을 컬러로 표시하여 경계가 명확히 보이도록 함.
    """
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Homography (현재 → 과거 방향)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                           matchesMask=matches_mask, flags=2)
        img_match = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good, None, **draw_params)
        cv2.imshow('ROI Feature Matching (Reverse)', img_match)

        # 현재 이미지를 과거 이미지 좌표계로 warp
        img2_warped = cv2.warpPerspective(img2_color, M, (img1_color.shape[1], img1_color.shape[0]))

        # ---- 경계선 표시 ----
        # 1. 과거 이미지 경계 (녹색)
        border_img1 = img1_color.copy()
        cv2.rectangle(border_img1, (0, 0), (img1_color.shape[1] - 1, img1_color.shape[0] - 1), (0, 255, 0), 1)
        # 2. warp된 현재 이미지 경계 (파란색)
        border_img2_warped = img2_warped.copy()
        cv2.rectangle(border_img2_warped, (0, 0), (img2_warped.shape[1] - 1, img2_warped.shape[0] - 1), (255, 0, 0), 1)

        # 3. Blend 이미지 (각 이미지의 경계 포함)
        blend_full = cv2.addWeighted(border_img1, 0.5, border_img2_warped, 0.5, 0)

        # ROI도 빨간색으로 표시
        x, y, w, h = roi
        cv2.rectangle(blend_full, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Blended Full Image (With Borders)', blend_full)
        return src_pts, dst_pts, M
    else:
        print("매칭점 부족")
        return None, None, None
    
def get_corners(shape):
    """이미지의 네 꼭짓점을 homogeneous 좌표로 반환"""
    h, w = shape[:2]
    return np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T  # shape: (3, 4)

def overlay_union_canvas(img_past, img_current, H, roi, alpha=0.5):
    # 1. 두 이미지의 네 꼭짓점 위치 구하기 (homogeneous 좌표)
    corners_past = get_corners(img_past.shape)
    corners_current = get_corners(img_current.shape)
    corners_current_warped = H @ corners_current
    corners_current_warped = (corners_current_warped[:2] / corners_current_warped[2:]).T  # (4,2)

    all_pts = np.vstack([corners_past[:2, :].T, corners_current_warped])
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    width = x_max - x_min
    height = y_max - y_min

    # 2. 새 캔버스(전체 이미지) 생성
    canvas_past = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_current = np.zeros_like(canvas_past)

    # 3. 과거 이미지를 canvas에 배치
    x_offset_past = -x_min
    y_offset_past = -y_min
    canvas_past[y_offset_past:y_offset_past + img_past.shape[0], x_offset_past:x_offset_past + img_past.shape[1]] = img_past

    # 4. 현재(warped) 이미지를 homography+offset 적용해서 canvas에 warp
    offset_mat = np.array([[1, 0, x_offset_past],
                           [0, 1, y_offset_past],
                           [0, 0, 1]])
    H_offset = offset_mat @ H
    img_current_warped = cv2.warpPerspective(img_current, H_offset, (width, height))
    canvas_current[:,:,:] = img_current_warped

    # 5. 알파 블렌딩 (addWeighted)
    blend = cv2.addWeighted(canvas_past, alpha, canvas_current, 1 - alpha, 0)

    # 6. 외곽선, ROI 강조
    # 과거 이미지 외곽(녹색)
    cv2.rectangle(blend, (x_offset_past, y_offset_past),
                  (x_offset_past + img_past.shape[1] - 1, y_offset_past + img_past.shape[0] - 1), (0,255,0), 1)
    # current(warped) 외곽(파란색)
    contour_pts = corners_current_warped + np.array([x_offset_past, y_offset_past])
    contour_pts = contour_pts.astype(int).reshape((-1, 1, 2))
    cv2.polylines(blend, [contour_pts], isClosed=True, color=(255,0,0), thickness=1)
    # ROI 사각형 (빨간색, past 이미지 기준)
    x, y, w, h = roi
    cv2.rectangle(blend, (x_offset_past + x, y_offset_past + y),
                  (x_offset_past + x + w, y_offset_past + y + h), (0,0,255), 1)

    cv2.imshow("Union Alpha Blend (addWeighted)", blend)
    return blend

def decompose_homography(H, K):
    """호모그래피를 분해하여 상대적 포즈(R, t)를 추정."""
    num, Rs, Ts, normals = cv2.decomposeHomographyMat(H, K)
    valid_poses = []
    for i in range(num):
        R = Rs[i]
        t = Ts[i]
        n = normals[i]
        # 카메라가 평면 앞에 있는지 확인 (z > 0)
        if t[2] > 0:  # 이동 벡터의 z 성분이 양수인지 확인
            valid_poses.append((R, t, n))
    return valid_poses

def evaluate_pose(pcd1, pcd2, R, t, visualize=True):
    """
    포즈의 정합성을 평가 (ICP를 사용해 점군 정합 오차 계산).
    ICP 전후 포인트 클라우드를 시각화.
    
    Args:
        pcd1: 과거 포인트 클라우드 (Open3D PointCloud)
        pcd2: 현재 포인트 클라우드 (Open3D PointCloud)
        R: 회전 행렬 (3x3 numpy array)
        t: 이동 벡터 (3x1 or 3, numpy array)
        visualize: 시각화 여부 (bool)
    
    Returns:
        fitness: ICP 정합 점수
        rmse: ICP 인라이어 RMSE
    """
    # 초기 변환 행렬 구성
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()  # t를 (3,1)에서 (3,)으로 변환
    
    # pcd1을 초기 포즈로 변환
    pcd1_transformed = deepcopy(pcd1)
    pcd1_transformed = pcd1_transformed.transform(T)
    
    # ICP 수행
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1_transformed, pcd2, max_correspondence_distance=0.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # ICP 후 변환 행렬 적용
    pcd1_icp_transformed = deepcopy(pcd1_transformed)
    pcd1_icp_transformed.transform(reg_p2p.transformation)
    
    if visualize:
        # 색상 설정
        pcd1_transformed.paint_uniform_color([1.0, 0.0, 0.0])  # 초기 변환 (빨강)
        pcd1_icp_transformed.paint_uniform_color([0.0, 1.0, 0.0])  # ICP 후 (녹색)
        pcd2.paint_uniform_color([0.0, 0.0, 1.0])  # 목표 (파랑)
        
        # ICP 전 시각화
        print("Visualizing ICP Before...")
        o3d.visualization.draw_geometries(
            [pcd1_transformed, pcd2],
            window_name="ICP Before",
            point_show_normal=False,
            width=1280,
            height=720
        )
        
        # ICP 후 시각화
        print("Visualizing ICP After...")
        o3d.visualization.draw_geometries(
            [pcd1_icp_transformed, pcd2],
            window_name="ICP After",
            point_show_normal=False,
            width=1280,
            height=720
        )
    
    return reg_p2p.fitness, reg_p2p.inlier_rmse

def evaluate_pose2(img1_color, roi, correspond_3d_points, correspond_3d_points_curr, src_pts, R, t, intrinsic, distortion, extrinsic, visualize=True):
    """
    재투영 오차로 포즈 평가.
    
    Args:
        correspond_3d_points: 과거 프레임의 3D 점 (Nx3 numpy array)
        correspond_3d_points_curr: 현재 프레임의 3D 점 (Nx3 numpy array)
        src_pts: 과거 이미지의 특징점 (Nx1x2 numpy array)
        R: 회전 행렬 (3x3 numpy array)
        t: 이동 벡터 (3x1 or 3, numpy array)
        intrinsic: 카메라 내부 파라미터 (3x3)
        distortion: 왜곡 계수
        extrinsic: 라이다 → 카메라 변환 (4x4)
        visualize: 시각화 여부 (bool)
    
    Returns:
        mean_error: 평균 재투영 오차 (픽셀)
        valid_ratio: 유효한 매칭 비율
    """
    errors = []
    src_pts_2d = src_pts.reshape(-1, 2)
    
    # 현재 3D 점을 과거 시점으로 변환
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    T_inv = np.linalg.inv(T)  # 과거→현재의 역변환 (현재→과거)
    
    for pt_3d_curr, pt_2d_past in zip(correspond_3d_points_curr, src_pts_2d):
        # 현재 3D 점을 과거 라이다 좌표계로 변환
        pt_3d_homo = np.append(pt_3d_curr, 1)  # 동차 좌표
        pt_3d_past = (T_inv @ pt_3d_homo)[:3]
        
        # 과거 카메라 좌표계로 변환 및 투영
        pt_3d_cam = (extrinsic @ np.append(pt_3d_past, 1))[:3]
        pt_2d, _ = cv2.projectPoints(pt_3d_cam.reshape(1, -1), np.zeros(3), np.zeros(3), intrinsic, distortion)
        pt_2d = pt_2d.squeeze()
        
        # 재투영 오차 계산
        error = np.linalg.norm(pt_2d - pt_2d_past)
        if error < 50:  # 이상치 제거 (임계값: 50픽셀)
            errors.append(error)
    
    mean_error = np.mean(errors) if errors else float('inf')
    valid_ratio = len(errors) / len(src_pts_2d) if len(src_pts_2d) > 0 else 0
    
    if visualize and errors:
        # 시각화: 과거 이미지에 투영된 점 표시
        img_vis = img1_color.copy()
        for pt_3d_curr, pt_2d_past in zip(correspond_3d_points_curr, src_pts_2d):
            pt_3d_homo = np.append(pt_3d_curr, 1)
            pt_3d_past = (T_inv @ pt_3d_homo)[:3]
            pt_3d_cam = (extrinsic @ np.append(pt_3d_past, 1))[:3]
            pt_2d, _ = cv2.projectPoints(pt_3d_cam.reshape(1, -1), np.zeros(3), np.zeros(3), intrinsic, distortion)
            pt_2d = pt_2d.squeeze().astype(int)
            cv2.circle(img_vis, tuple(pt_2d), 5, (0, 255, 0), 2)  # 투영 점 (녹색)
            cv2.circle(img_vis, tuple(pt_2d_past.astype(int)), 5, (0, 0, 255), 2)  # 원래 점 (빨강)
            cv2.line(img_vis, tuple(pt_2d), tuple(pt_2d_past.astype(int)), (255, 0, 0), 1)  # 오차 선 (파랑)
        cv2.rectangle(img_vis, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 0, 255), 2)
        cv2.imshow(f"Reprojection Error (Mean: {mean_error:.2f} pixels)", img_vis)
    
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
    valid = (pcd_cam[:,2] > 0) \
        & (uv[:,0] >= 0) & (uv[:,0] < img_shape[1]) \
        & (uv[:,1] >= 0) & (uv[:,1] < img_shape[0])
    uv_valid = uv[valid]
    pcd_valid = pcd_points[valid]
    return uv_valid, pcd_valid

def match_2d_to_3d(src_pts, uv_valid, pcd_valid):
    src_pts_2d = src_pts.reshape(-1, 2)
    tree2d = cKDTree(uv_valid)
    dist, idx = tree2d.query(src_pts_2d, distance_upper_bound=3)
    valid_mask = dist != np.inf
    matched_3d = pcd_valid[idx[valid_mask]]
    valid_indices = np.where(valid_mask)[0]  # src_pts에서 유효한 점의 인덱스
    return matched_3d, dist, idx, valid_indices

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

def main():
    # File paths
    img1_path = '/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/camera1_image_color_compressed/2025-05-16_103546536693.jpg'
    img2_path = '/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-05-17/cam1/2025-05-17_13-03-50.png'
    pcd1_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-24/ouster1_points/2025-05-16_103546536693.pcd"
    pcd2_path = "0424-0515/2025-05-17/os1/2025-05-17_13-03-50.pcd"
    intrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/intrinsic1.csv"
    extrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/calibration/iou_optimized_transform1.txt"
    roi = (500, 50, 300, 130)  # (x, y, w, h)

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

    # Homography computation and visualization (reverse)
    src_pts, dst_pts, M_inv = compute_homography_and_visualize_reverse(img1_color, img2_color, kp1, kp2, good, roi)
    if src_pts is None:
        return

    print("Homography Matrix:\n", M) # past → current
    print("Homography Matrix:\n", M_inv) # current → past
    overlay_union_canvas(img1_color, img2_color, M_inv, roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Point cloud processing - Past data    
    pcd_points = np.asarray(pcd1.points)
    uv, pcd_cam = project_points_to_image(pcd_points, intrinsic, distortion, extrinsic)
    uv_valid, pcd_valid = filter_valid_points(uv, pcd_points, pcd_cam, img1_color.shape)
    correspond_3d_points, dist, idx, valid_idx = match_2d_to_3d(src_pts, uv_valid, pcd_valid)
    #print(idx, valid_idx)
    correspond_3d_pcd = o3d.geometry.PointCloud()
    correspond_3d_pcd.points = o3d.utility.Vector3dVector(correspond_3d_points)
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
    correspond_3d_points_curr, dist_curr, idx_curr, valid_idx_curr = match_2d_to_3d(dst_pts, uv_valid_curr, pcd_valid_curr)
    #print(idx_curr, valid_idx_curr)
    correspond_3d_pcd_curr = o3d.geometry.PointCloud()
    correspond_3d_pcd_curr.points = o3d.utility.Vector3dVector(correspond_3d_points_curr)
    # Visualize 3D points
    visualize_3d_points(pcd2, correspond_3d_points_curr, extrinsic_inv)
    # Visualize 2D matches
    visualize_2d_matches(img2_color, dst_pts, uv_valid_curr, dist_curr, roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    if M is not None:
        # 호모그래피 분해
        valid_poses = decompose_homography(M, intrinsic)
        print(f"Found {len(valid_poses)} valid pose solutions")
        for i, (R, t, n) in enumerate(valid_poses):
            print(f"Pose {i+1}:")
            print("Rotation Matrix (R):\n", R)
            print("Translation Vector (t):\n", t)
            print("Plane Normal (n):\n", n)

        if valid_poses:
            # 일치하는 특징점-3D 점 쌍 필터링 (질문자의 코드 적용)
            # 과거 correspondence (매칭 성공한 것만)
            valid_mask_past = (dist != np.inf)
            # 현재 correspondence (매칭 성공한 것만)
            valid_mask_now = (dist_curr != np.inf)

            # 매칭된 전체 인덱스는 FLANN good match 개수 기준
            assert len(valid_mask_past) == len(valid_mask_now)  # 둘 다 good match 개수와 동일

            # 1:1로 대응되는 인덱스만 남기기 (둘 다 성공한 것만)
            joint_mask = valid_mask_past & valid_mask_now

            # 3D correspondence 생성
            matched_3d_curr = []  # 질문자의 A_list (현재 3D 점)
            matched_3d_past = []  # 질문자의 B_list (과거 3D 점)
            matched_src_pts = []
            N = len(src_pts)  # good match 개수

            for i in range(N):
                if joint_mask[i]:
                    # pcd_valid와 idx로 바로 접근
                    matched_3d_curr.append(pcd_valid_curr[idx_curr[i]])  # 현재 3D 점
                    matched_3d_past.append(pcd_valid[idx[i]])            # 과거 3D 점
                    matched_src_pts.append(src_pts[i])                   # 과거 2D 점

            # Numpy array로 변환
            matched_3d_curr = np.array(matched_3d_curr)
            matched_3d_past = np.array(matched_3d_past)
            matched_src_pts = np.array(matched_src_pts)
            print("최종 correspondence 쌍 개수:", len(matched_3d_past))

            if len(matched_3d_past) < 4:
                print("Insufficient matched 3D points for evaluation")
                return

            # 포즈 평가
            best_pose = None
            best_fitness = float('inf')  # 최소 오차를 찾으므로 inf로 초기화
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
                # 호모그래피 방향 보정 (현재→과거를 과거→현재로)
                R = R.T
                t = -R @ t
                # 스케일 보정
                def estimate_plane_distance(pcd, extrinsic):
                    points = np.asarray(pcd.points)
                    pcd_cam = (extrinsic @ np.hstack([points, np.ones((points.shape[0], 1))]).T).T[:, :3]
                    return np.mean(pcd_cam[:, 2])
                d = estimate_plane_distance(pcd1, extrinsic)
                t = t * d
                print("Best Pose from Homography (Past→Current):")
                print("Rotation Matrix (R):\n", R)
                print("Translation Vector (t):\n", t)
                print("Plane Normal (n):\n", n)
                relative_pose = np.eye(4)   
                relative_pose[:3, :3] = R
                relative_pose[:3, 3] = t.flatten()

                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).transform(extrinsic)
                cam_frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3).transform(relative_pose)
                lidar_frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]).transform(relative_pose @ extrinsic)
                lidar_frame.paint_uniform_color([0.0, 1.0, 0.0])   # 과거 카메라 좌표계 (녹색)
                lidar_frame_est.paint_uniform_color([0.0, 0.0, 1.0])  # 현재 카메라 좌표계 (파란색)
                cam_frame.paint_uniform_color([0.0, 1.0, 0.0])  # 과거 카메라 좌표계 (녹색)
                cam_frame_est.paint_uniform_color([0.0, 0.0, 1.0])  # 현재 카메라 좌표계 (파란색)

                o3d.visualization.draw_geometries([cam_frame, cam_frame_est, lidar_frame, lidar_frame_est],)
                pcd1_temp = pcd1.transform(extrinsic)
                pcd2_temp = pcd2.transform(relative_pose @ extrinsic)
                pcd1.paint_uniform_color([0.7, 0.7, 0.7])
                pcd2_temp.paint_uniform_color([0.0, 1.0, 0.0])  # 현재 포인트 클라우드 (녹색)

                o3d.visualization.draw_geometries([pcd1_temp, pcd2_temp, lidar_frame, cam_frame, cam_frame_est],
                                                  point_show_normal=False, width=1280, height=720,
                                                  window_name='Matched Points Visualization')
    



if __name__ == "__main__":
    main()