import cv2
import numpy as np

def get_vertical_lines(image):
    """
    通过边缘检测和霍夫变换提取图像中近似竖直的直线
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 边缘检测 (参数可根据图像清晰度微调)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 霍夫变换提取直线
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                            minLineLength=80, maxLineGap=10)
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线的倾斜角度
            if x2 == x1:
                angle = 90.0
            else:
                angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            
            # 过滤掉水平线和倾斜线，只保留角度大于 70 度的近似竖直线
            if angle > 70:
                vertical_lines.append((x1, y1, x2, y2))
                
    return vertical_lines

def compute_vanishing_point(lines):
    """
    计算所有竖直线的交点，并取中位数以剔除噪点（求垂直灭点）
    """
    intersections = []
    # 遍历所有线对，计算交点
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i]
            x3, y3, x4, y4 = lines[j]
            
            # 利用齐次坐标叉乘求直线方程
            L1 = np.cross([x1, y1, 1], [x2, y2, 1])
            L2 = np.cross([x3, y3, 1], [x4, y4, 1])
            
            # 直线相交点
            pt = np.cross(L1, L2)
            if pt[2] != 0: # 避免平行线导致除以 0
                pt = pt / pt[2]
                intersections.append([pt[0], pt[1]])
                
    if not intersections:
        return None
        
    # 取所有交点的中位数作为鲁棒的灭点坐标 (vp_x, vp_y)
    # 中位数对异常值(误检的线条)有极强的抵抗力
    vp = np.median(intersections, axis=0)
    return vp

def auto_rectify_building(image_path, output_path="rectified_auto_东南.jpg"):
    """
    自动化调正视角主函数
    """
    img = cv2.imread(image_path)
    if img is None:
        print("未找到图片，请检查路径。")
        return
        
    h, w = img.shape[:2]
    
    # 1. 获取竖直线
    lines = get_vertical_lines(img)
    if len(lines) < 2:
        print("检测到的竖直线不足，无法自动对齐。")
        return
        
    # 2. 计算垂直灭点
    vp = compute_vanishing_point(lines)
    if vp is None:
        print("无法计算灭点。")
        return
        
    vp_x, vp_y = vp
    print(f"计算得到的垂直灭点坐标: X={vp_x:.2f}, Y={vp_y:.2f}")

    # 3. 构造矫正矩阵 (Homography)
    # 思想：将图像中心移到原点 -> 应用校正映射 -> 移回原点
    cx, cy = w / 2.0, h / 2.0
    
    # 计算灭点相对于中心的偏移
    vp_y_centered = vp_y - cy
    
    # 构造透视变换矩阵，专门修正俯仰角(Pitch)导致的畸变
    # (如果图像没有明显的左右倾斜，仅保留对 Y 轴的修正即可消除"远小近大")
    K = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, -1.0 / vp_y_centered, 1]
    ])
    
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])
    
    T2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ])
    
    # H = T2 * K * T1
    H = T2.dot(K).dot(T1)

    # 4. 调整输出画布大小，防止矫正后图像被裁剪
    # 计算原图四个角在变换后的坐标
    corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype='float32')
    corners = np.array([corners])
    warped_corners = cv2.perspectiveTransform(corners, H)[0]
    
    x_min, y_min = np.int32(warped_corners.min(axis=0))
    x_max, y_max = np.int32(warped_corners.max(axis=0))
    
    # 计算新图像的尺寸和偏移
    new_w = x_max - x_min
    new_h = y_max - y_min
    
    # 构造平移矩阵，把画面移回正中央
    T_offset = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    H_final = T_offset.dot(H)

    # 5. 应用变换并保存
    rectified_img = cv2.warpPerspective(img, H_final, (new_w, new_h))
    
    cv2.imwrite(output_path, rectified_img)
    print(f"处理完成！图片已保存至: {output_path}")

# ================= 使用示例 =================
# 将 "your_image.jpg" 替换为你的输入图像路径
if __name__ == "__main__":
    auto_rectify_building("../data/southwest.jpg", "rectified_auto_southwest.jpg")