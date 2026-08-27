import cv2,os
import numpy as np

def get_vertical_lines(image):
    """提取近似垂直的线段"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                            minLineLength=80, maxLineGap=10)
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = 90.0 if x2 == x1 else np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle > 70:
                vertical_lines.append((x1, y1, x2, y2))
    return vertical_lines

def compute_vanishing_point(lines):
    """利用中位数求鲁棒的垂直交点 (灭点)"""
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i]
            x3, y3, x4, y4 = lines[j]
            L1 = np.cross([x1, y1, 1], [x2, y2, 1])
            L2 = np.cross([x3, y3, 1], [x4, y4, 1])
            pt = np.cross(L1, L2)
            if pt[2] != 0:
                pt = pt / pt[2]
                intersections.append([pt[0], pt[1]])
    if not intersections: return None
    return np.median(intersections, axis=0)

def physically_accurate_rectify(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("未找到图片，请检查路径。")
        return
        
    h, w = img.shape[:2]
    
    # 1. 计算灭点
    lines = get_vertical_lines(img)
    if len(lines) < 2:
        print("检测到的竖直线不足。")
        return
    vp = compute_vanishing_point(lines)
    if vp is None: return
    vp_x, vp_y = vp

    # 2. 核心修正：基于物理相机的焦距估算
    # 经验上，手机或常规相机的焦距约等于图像对角线长度
    f = np.sqrt(w**2 + h**2) 
    cx, cy = w / 2.0, h / 2.0
    
    # 构造相机内参矩阵 K
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
    ])
    K_inv = np.linalg.inv(K)
    
    # 3. 计算相机的俯仰角 (Pitch)
    # 利用灭点到图像中心的距离与焦距的三角关系
    theta = np.arctan((cy - vp_y) / f)
    
    # 构造绕 X 轴的物理旋转矩阵 (抵消仰视角度)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    
    # 4. 生成保持真实长宽比的透视矩阵
    H = K @ R_x @ K_inv

    # ================= 以下为动态画布与裁剪逻辑 =================
    corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype='float32').reshape(-1, 1, 2)
    
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    
    x_min, y_min = np.int32(np.floor(warped_corners.min(axis=0)))
    x_max, y_max = np.int32(np.ceil(warped_corners.max(axis=0)))
    new_w, new_h = x_max - x_min, y_max - y_min
    
    T_offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H_final = T_offset @ H

    # 执行透视变换
    rectified_img = cv2.warpPerspective(img, H_final, (new_w, new_h))

    # 计算裁剪边界：找内接矩形，切除所有黑边
    final_corners = cv2.perspectiveTransform(corners, H_final).reshape(-1, 2)
    TL, TR, BR, BL = final_corners

    crop_left = max(0, int(max(TL[0], BL[0])))
    crop_right = min(new_w, int(min(TR[0], BR[0])))
    crop_top = max(0, int(max(TL[1], TR[1])))
    crop_bottom = min(new_h, int(min(BL[1], BR[1])))

    # 安全检查：确保裁剪框有效
    if crop_left < crop_right and crop_top < crop_bottom:
        final_img = rectified_img[crop_top:crop_bottom, crop_left:crop_right]
    else:
        print("警告：形变过大导致无法安全裁剪，输出带黑边的全图。")
        final_img = rectified_img

    cv2.imwrite(output_path, final_img)
    print(f"处理完成！完美比例且无黑边的图片已保存至: {output_path}")

if __name__ == "__main__":
    for file_name in os.listdir("../data"):
        if file_name.endswith(".jpg"):
            physically_accurate_rectify(f"../data/{file_name}", os.path.join("../data", f"rectified_{file_name}"))