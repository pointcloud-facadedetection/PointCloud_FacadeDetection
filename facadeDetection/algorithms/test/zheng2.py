import cv2
import numpy as np
import os

def get_vertical_lines(image):
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

def auto_rectify_and_crop(image_path, output_path="rectified_cropped.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print("未找到图片，请检查路径。")
        return
        
    h, w = img.shape[:2]
    
    lines = get_vertical_lines(img)
    vp = compute_vanishing_point(lines)
    if vp is None: return
    vp_x, vp_y = vp

    cx, cy = w / 2.0, h / 2.0
    vp_y_centered = vp_y - cy
    
    K = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, -1.0 / vp_y_centered, 1]
    ])
    
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    H = T2.dot(K).dot(T1)

    # 计算原图四个角在初始变换矩阵 H 下的坐标
    # 顺序：左上(TL), 右上(TR), 右下(BR), 左下(BL)
    corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype='float32').reshape(-1, 1, 2)
    
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    
    # 计算包含整个梯形的外部画布尺寸和偏移量
    x_min, y_min = np.int32(np.floor(warped_corners.min(axis=0)))
    x_max, y_max = np.int32(np.ceil(warped_corners.max(axis=0)))
    new_w = x_max - x_min
    new_h = y_max - y_min
    
    T_offset = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    H_final = T_offset.dot(H)

    # 应用最终变换生成带有黑边的大图
    rectified_img = cv2.warpPerspective(img, H_final, (new_w, new_h))

    # ================= 新增自动裁剪逻辑 =================
    # 利用最终矩阵 H_final 计算原图四个角在新图上的精准位置
    final_corners = cv2.perspectiveTransform(corners, H_final).reshape(-1, 2)
    TL_f, TR_f, BR_f, BL_f = final_corners

    # 寻找最大内接矩形（避开左右侧的黑色三角区域）
    # 左边界：左上和左下角点中 X 坐标偏右的那一个
    crop_left = int(max(TL_f[0], BL_f[0]))
    # 右边界：右上和右下角点中 X 坐标偏左的那一个
    crop_right = int(min(TR_f[0], BR_f[0]))
    # 上下边界同理
    crop_top = int(max(TL_f[1], TR_f[1]))
    crop_bottom = int(min(BL_f[1], BR_f[1]))

    # 安全越界保护
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(new_w, crop_right)
    crop_bottom = min(new_h, crop_bottom)

    # 执行 NumPy 数组切片完成裁剪
    cropped_img = rectified_img[crop_top:crop_bottom, crop_left:crop_right]
    # ====================================================

    cv2.imwrite(output_path, cropped_img)
    print(f"处理完成！无黑边图片已保存至: {output_path}")

if __name__ == "__main__":
    for file_path in os.listdir("../data"):
        if file_path.endswith(".jpg"):
            auto_rectify_and_crop(os.path.join("../data", file_path), os.path.join("../data/guizheng", f"rectified_cropped_{file_path}"))
