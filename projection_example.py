import cv2
import numpy as np
import os
import open3d as o3d

def pcd_projection(img,
                   pcd,
                   intrinsic,
                   distortion,
                   transform,
                   point_size=3,
                   color=None,
                   num_bins=6,
                   colormap=cv2.COLORMAP_TURBO,
                   gamma=0.9):
    """
    LiDAR 포인트 클라우드를 카메라 영상에 투영하고,
    거리별(깊이별) 색상을 계단식으로 부여한다.

    Parameters
    ----------
    img : np.ndarray (H, W, 3, BGR)
    pcd : open3d.geometry.PointCloud
    intrinsic : (3,3) np.ndarray
    distortion : (N,) np.ndarray
    transform : (4,4) np.ndarray
    point_size : int, plotted circle radius
    color : None 혹은 (3,) 또는 (N,3) uint8  [B,G,R]
    num_bins : int, 거리 구간 수 (색 구분 단계)
    colormap : OpenCV colormap ID (ex. cv2.COLORMAP_TURBO)
    gamma : float, 0 < gamma ≤ 3, 1보다 작으면 가까운 거리 강조

    Returns
    -------
    img_out : np.ndarray
    valid_points : (M,2) np.ndarray  이미지 좌표
    """
    if not pcd.has_points():
        return img, []

    # ── 1. 클라우드 변환 ───────────────────────────────────────────────
    pts = np.asarray(pcd.points)
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    pts_cam = (transform @ pts_h.T).T[:, :3]

    # 카메라 앞쪽 Z>0
    mask_front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask_front]
    if pts_cam.size == 0:
        return img, []

    # ── 2. 거리 계산 및 계단식 정규화 ─────────────────────────────────
    dist = np.linalg.norm(pts_cam, axis=1)
    # gamma 보정으로 근거리 분해능 향상
    dist_gamma = dist ** gamma

    # 0~255 정규화 후 binning
    dist_norm = cv2.normalize(dist_gamma, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # 계단식 효과: 0~255를 num_bins 구간으로 나눠 중앙값으로 스냅
    step = 256 // num_bins
    dist_quant = (dist_norm // step) * step + step // 2
    colors_map = cv2.applyColorMap(dist_quant, colormap)

    # ── 3. 사용자 지정 색상 우선 적용 ────────────────────────────────
    if color is None:
        colors = colors_map
    else:
        color = np.asarray(color, dtype=np.uint8)
        colors = np.tile(color, (len(pts_cam), 1)) if color.ndim == 1 else color

    # ── 4. 3‑D → 2‑D 투영 ────────────────────────────────────────────
    proj, _ = cv2.projectPoints(pts_cam,
                                rvec=np.zeros((3, 1), np.float32),
                                tvec=np.zeros((3, 1), np.float32),
                                cameraMatrix=intrinsic,
                                distCoeffs=distortion)
    proj = proj.reshape(-1, 2)

    h, w = img.shape[:2]
    in_img = (proj[:, 0] >= 0) & (proj[:, 0] < w) & \
             (proj[:, 1] >= 0) & (proj[:, 1] < h)

    pts_2d = proj[in_img].astype(int)
    pts_color = colors[in_img]

    # ── 5. 렌더링 ────────────────────────────────────────────────────
    for (x, y), c in zip(pts_2d, pts_color):
        bgr = tuple(int(v) for v in c.squeeze())   # c: (1,3) → (3,)
        cv2.circle(img, (x, y), point_size, bgr, -1)

    return img, pts_2d

if __name__ == '__main__':
    # 경로 지정
    image_path = "/home/myungw00/ROS/gm/Code/data/scripts/results/projection images/sample/cam1_2025-05-16_103546536693.jpg"
    pointcloud_path = "/home/myungw00/ROS/gm/Code/data/scripts/results/projection images/sample/ouster1_roi2.pcd"
    result_path = "/home/myungw00/ROS/gm/Code/data/scripts/results/projection images/sample"
    intrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/results/projection images/sample/intrinsic1.csv"
    extrinsic_path = "/home/myungw00/ROS/gm/Code/data/scripts/results/projection images/sample/iou_optimized_transform1.txt"



    # 이미지 파일 1개만 사용
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # PCD 파일 1개 불러오기
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    # PCD에 상대 변환 적용

    # 카메라 파라미터 로딩
    intrinsic_param = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
    intrinsic = np.array([
        [intrinsic_param[0]/2, intrinsic_param[1], intrinsic_param[2]/2],
        [0.0, intrinsic_param[3]/2, intrinsic_param[4]/2],
        [0.0, 0.0, 1.0]
    ])
    distortion = np.array(intrinsic_param[5:])

    extrinsic = np.loadtxt(extrinsic_path, delimiter=',')
    if extrinsic.shape == (3, 4):
        # 3x4 행렬인 경우 4x4로 변환
        ext_tmp = np.eye(4)
        ext_tmp[:3, :] = extrinsic
        extrinsic = ext_tmp

    print("Camera Matrix:\n", intrinsic)
    print("Distortion Coefficients:\n", distortion)
    print("Extrinsic Matrix:\n", extrinsic)

    optimized_image, _ = pcd_projection(
        image.copy(),
        pcd,
        intrinsic,
        distortion,
        extrinsic,
        point_size=1,
        color=None,
        num_bins=10,
        colormap=cv2.COLORMAP_TURBO,
        gamma=1
    )
    # 결과 이미지 저장
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, "image_pcd.jpg")
    cv2.imwrite(save_path, optimized_image)
    print(f"Projected image saved at: {save_path}")
