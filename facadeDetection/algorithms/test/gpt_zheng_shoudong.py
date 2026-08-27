import cv2
import numpy as np


# ============================================================
# 1. 四点排序
# 顺序统一为：
#
#   TL -------- TR
#   |            |
#   |            |
#   BL -------- BR
#
# ============================================================
def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    # x + y
    s = pts.sum(axis=1)

    # y - x
    diff = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


# ============================================================
# 2. 根据四个点计算 Homography，并进行透视矫正
#
# aspect_ratio:
#     None：自动估算输出尺寸
#     如果知道真实建筑宽高比，例如 width / height = 0.45，
#     可以设置 aspect_ratio=0.45
# ============================================================
def rectify_perspective(image, pts, aspect_ratio=None):

    rect = order_points(pts)

    tl, tr, br, bl = rect

    # --------------------------------------
    # 估计输出宽度
    # --------------------------------------
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    max_width = int(max(width_top, width_bottom))

    # --------------------------------------
    # 估计输出高度
    # --------------------------------------
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    max_height = int(max(height_left, height_right))

    # 如果知道真实的宽高比，最好使用真实比例
    if aspect_ratio is not None:
        max_height = int(max_width / aspect_ratio)

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    # --------------------------------------
    # 矫正后的四个目标点
    #
    # (0,0) ---------------- (W,0)
    #   |                       |
    #   |                       |
    # (0,H) ---------------- (W,H)
    # --------------------------------------
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # --------------------------------------
    # 求 Homography
    #
    # x_rectified ~ H * x_original
    # --------------------------------------
    H = cv2.getPerspectiveTransform(rect, dst)

    # --------------------------------------
    # 透视变换
    # --------------------------------------
    warped = cv2.warpPerspective(
        image,
        H,
        (max_width, max_height)
    )

    return warped, H


# ============================================================
# 3. 鼠标选四个点
# ============================================================
clicked_points = []
display_image = None
scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(clicked_points) >= 4:
            return

        clicked_points.append((x, y))

        # 绘制点击位置
        cv2.circle(
            display_image,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display_image,
            str(len(clicked_points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # 连线，方便检查
        if len(clicked_points) > 1:
            cv2.line(
                display_image,
                clicked_points[-2],
                clicked_points[-1],
                (0, 255, 0),
                2
            )

        if len(clicked_points) == 4:
            cv2.line(
                display_image,
                clicked_points[-1],
                clicked_points[0],
                (0, 255, 0),
                2
            )

        cv2.imshow("Select facade", display_image)


# ============================================================
# 4. 主程序
# ============================================================
if __name__ == "__main__":

    image_path = "../data/southeast.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    H_img, W_img = image.shape[:2]

    # --------------------------------------------------------
    # 图片太大时缩小显示
    # 只影响交互窗口，不影响最终计算精度
    # --------------------------------------------------------
    max_display_height = 1000
    max_display_width = 1400

    scale = min(
        max_display_height / H_img,
        max_display_width / W_img,
        1.0
    )

    display_image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale
    )

    original_display = display_image.copy()

    print("")
    print("======================================")
    print("建筑立面透视矫正")
    print("======================================")
    print("请在同一个建筑立面上点击4个点。")
    print("")
    print("建议点击一个实际为矩形的区域的四角。")
    print("")
    print("操作：")
    print("  鼠标左键：选择点")
    print("  r：重新选择")
    print("  Enter：完成")
    print("  ESC：退出")
    print("======================================")
    print("")

    cv2.namedWindow(
        "Select facade",
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        "Select facade",
        mouse_callback
    )

    while True:

        cv2.imshow(
            "Select facade",
            display_image
        )

        key = cv2.waitKey(20) & 0xFF

        # r：重置
        if key == ord("r"):
            clicked_points = []
            display_image = original_display.copy()
            print("Points reset.")

        # Enter：开始矫正
        elif key == 13:

            if len(clicked_points) != 4:
                print(
                    f"当前只有 {len(clicked_points)} 个点，请选择4个点。"
                )
                continue

            break

        # ESC：退出
        elif key == 27:
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # 将显示图上的坐标恢复到原始分辨率
    # --------------------------------------------------------
    pts = np.array(
        clicked_points,
        dtype=np.float32
    )

    pts = pts / scale

    print("\nSelected points in original image:")
    print(pts)

    # --------------------------------------------------------
    # 执行透视矫正
    # --------------------------------------------------------
    rectified, H_rectify = rectify_perspective(
        image,
        pts,
        aspect_ratio=None
    )

    print("\nHomography H:")
    print(H_rectify)

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    cv2.imwrite(
        "rectified.jpg",
        rectified
    )

    np.save(
        "H_rectify.npy",
        H_rectify
    )

    print("\nSaved:")
    print("  rectified.jpg")
    print("  H_rectify.npy")

    # --------------------------------------------------------
    # 显示结果
    # --------------------------------------------------------
    cv2.namedWindow(
        "Rectified facade",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Rectified facade",
        rectified
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()