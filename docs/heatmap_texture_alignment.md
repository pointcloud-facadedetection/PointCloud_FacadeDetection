# 复核页热力贴图方向一致性验证记录

日期：2026-10-15。结论：**贴图方向与算法实际面方向一致，无镜像/翻转/轴互换**。

## 实证验证（csxm 项目真实数据）

1. **PNG 内容 ↔ 立面 UV**：质量报告 1996 个失败靠尺窗口中心投影到
   `facade_001_ruler_flatness_area_transparent.png`，正常方向命中率 98.6%；
   v 翻转 39.2%、uv 全翻转 33.7%（基准不透明率 37%）。方向唯一确定且正确。
2. **VTK 纹理约定**：2×2 四角四色纹理实测——`t=0` 对应 numpy 底行、`t=1` 顶行
   （pyvista `Texture._from_array` 内部做 `flip(swapaxes)`，见 texture.py:264-268）。
   现 TCoords `[[0,0],[1,0],[1,1],[0,1]]`（ruiqi 行序修正版）使 v_min 角采样
   PNG 底行（v_min 内容）✓。修正前的版本是镜像的。
3. **法向朝向**：facade 43/44 的 `normal·(u×v)=+1.000`（CCW 一致），法向朝点云
   外侧/扫描侧（+11.86/+4.46），相机正视落在贴图正面，不会看到背面镜像。

## 公式链一致性

「导出 PNG」与「视口贴图」两条路径的 `像素→UV→世界坐标` 逐点一致：
行=v/列=u 语义在 rasterize（projection.py:99-100）、裁剪
（heatmap_renderer.py:875-900）、重建（interactive_ruler.py:190-262）、
贴图（pyvista_viewport.py:1161-1233）四处统一。

## 遗留风险（非现行 bug，备查）

1. `facade_geometry.py:9-11` docstring 与实现矛盾：声称规范 v 轴与 +Z 同向，
   实际按手性翻 u（:59-61），且 `up_in_plane`（:54-57）是死代码。该分支实践中
   永不触发（plane_axes 两分支恒有 u×v·n>0），但会误导维护者。
2. 贴图轴架取自 `facade_geometry`、UV 矩形取自 projection 坐标架：今天两者
   相等；若手性翻转分支被触发（geom_u = −proj_u），贴图会水平镜像。
   回退路径（重建失败时用 domain 矩形，inspection_review_page.py:347-348）
   无 5% 外扩与 mask 裁剪，会有轻微错位。
3. 重建未复刻 rasterize 的缺陷点外扩（projection.py:78-82）；实际缺陷点都在
   domain+5% 内不触发，万一触发由尺寸守卫兜住回退。
4. ±1 像素边界约定：quad 比真实物理足迹右/底各窄 1px（≈1cm），视觉不可察觉。
5. 重建用 float32 投影（导出用 float64），极端坐标下可能差 1 像素裁剪列，
   由尺寸守卫兜住。
