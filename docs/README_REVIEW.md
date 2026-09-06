# README / ARCHITECTURE 文档审查

> 基线：`charles_optimize` @ `44e5c5d`（已含最新 `origin/ruiqi_merge`）
> 审查对象：`README.md` 与 `docs/ARCHITECTURE.md`
> 日期：2026-09-03

## 〇、先说一个管理问题

`README.md` 与 `docs/ARCHITECTURE.md` 内容几乎完全相同，仅"services 负责人"两处署名不同（`ruiqi / haolin` vs `haolin`）——两份副本**已经开始漂移**。建议：README 只留项目简介 + 快速上手，架构细节收敛到 `docs/ARCHITECTURE.md` 一份，README 链接过去。

---

## 一、README 本身的问题（文档层面）

1. **目录树是初期示意图，与现实脱节**
   - 所有条目标注"（示意）"，且把 `ui/`、`view3d/`、`db/`、`models/`、`algorithms/`、`services/`、`config/`、`utils/` 画在仓库根，实际全部位于 `facadeDetection/` 包内。

2. **引用的文件不存在**
   - `db/migrations/`（Alembic 迁移）：不存在。实际迁移是 `facadeDetection/db/connection.py:103-114` 的运行时 `ALTER TABLE` 补丁。
   - `models/pointcloud.py`、`models/analysis.py`、`models/registration.py`：均不存在。实际对应物为 `file_asset.py`（点云资产）、`facade.py`（Facade/QualityMetric/Heatmap）、`pointcloud_station.py`（配准记录挂在 station 表）。
   - `ui/resources/`：不存在。
   - 示例引用 `facadeDetection/services/facade_service.py` 已变为子包 `services/facade/facade_service.py`。

3. **入口描述错误，且无运行说明**
   - README 称 `main.py` 为应用入口；实际根目录 `main.py` 是 Flask Web 服务入口，桌面端入口是 `facadeDetection/main.py`。
   - 桌面端必须从 `facadeDetection/` 目录启动（包内是 `from ui.main_window import ...` 裸导入），README 只字未提。
   - `facadeDetection/requirements.txt` 已存在（含 `qtwebview2`、`pypinyin` 等不显然的依赖），README 无环境/依赖章节。

4. **技术选型描述与实现不符**
   - README 称用 Open3D `visualization.gui` widget 嵌入 PySide6；实际是**原生 Open3D Visualizer 窗口 + Win32 HWND 查找嵌入**（`view3d/window_embed.py`，Windows 用 ctypes `EnumWindows`，Linux 依赖 xdotool/wmctrl，macOS 直接不支持）。这是关键架构约束，文档未写。

5. **分层图缺层**
   - 无 `services/dal/`（仓储层）、无 `runtime/`（lifecycle / task_context / task_scheduler，横切层）。
   - `view3d/` 实际有 8 个模块（scene / camera / interaction / roi_selection / pick_overlay / lod / geometry_factory / open3d_adapter），README 只列 2 个。

6. **完全未提存储设计**
   - 实际是双库结构：全局 `index.db` + per-project `project.db`，外加 `pcfd/index.json` 项目索引和 `Storage` 目录解析（`FACD_DATA_DIR` 环境变量 > 仓库 `data/` > `%LOCALAPPDATA%`）。新人无法从文档理解数据落在哪。

7. **模块分工表只到目录级**，且与现有模块对不上（如 `algorithms/` 写"检测、分割、配准、质量评估"，实际分割在 Web 端，桌面端是 FLS 管线、照片匹配等）。

---

## 二、实现超出 README 的部分（文档未覆盖的实际功能）

1. **双应用并存**：整个 Web 应用（`backend/`、`app.py`、根 `main.py`、`templates/index.html` 5399 行）README 完全未提。按"Web 仅保留"的口径，README 至少应声明其存在与定位（遗留/冻结，不再投入）。

2. **四页工作台**：项目概览 / 项目操作 / 检测复核 / 报告预览，无边框窗口 + 自绘标题栏 + 底部 Dock。README 只有一句"主窗口（示意）"。

3. **多站点管理与 ICP 配准**：`pointcloud_station_service`、`pointcloud_stations` 表、视图状态持久化（`pointcloud_view_states`）、多尺度点到面 ICP（`algorithms/registration/`）。

4. **FLS 数据管线**：`.dist` 距离文件读取（`utils/dist_reader.py`）、距离分层下采样（`algorithms/geometry.py:278` `stratified_proxy_build`）、`VoxelCascadeIndex` 三级索引（voxel→raw / proxy→source CSR 双域映射）、FlsRead.exe 子进程转换（外部 exe 依赖，文档未声明）。

5. **质量评估体系**：2m 靠尺法平整度/垂直度（`algorithms/facade/ruler_quality.py`，ProcessPool 并行）、检测标准 profile（`services/inspection_profile.py`，GB 50204/50210 六个预设）、质量结果持久化（`facades` / `quality_metrics` 表 + npz artifact）。

6. **报告导出已实现**（最新合入）：`services/report_export/pdf_report_renderer.py`（589 行）+ `report_data_service.py`，已接入 `main_window.py`。README 连"报告"概念都没有（`models/report.py`、`reports` 表同样无文档）。

7. **热力图体系**：`result_export_service.py` PNG 导出（cv2 栅格化 + 图例）、`heatmap_spec.py`、`algorithms/facade/projection.py`。

8. **照片-点云匹配算法包**：`algorithms/photo_pointcloud_matching/`（PnP 求解 668 行 + 热力图 729 行）。**桌面端无任何调用方，仅 backend Web 端在用**——代码位置与使用方错位，README 未提。

9. **项目创建细节**：省市区三级联动（`utils/pca-code.json`）、项目目录拼音缩写命名（`pypinyin`）——影响存储布局理解。

---

## 三、README 写了、但实现没做到或与约定相悖

1. **"services 四步模式 + 构造函数注入"大体遵守，但有破口**：
   - service 层弹窗：`ProjectOperationService` import 并使用 `QMessageBox` / `QColorDialog`（`project_operation_service.py:5`），违反 UI→services 单向语义。
   - UI 绕过 service 直连 DAL：`main_window.py:1676-1685` 直接调 `ResultsRepo.persist_quality_artifact / commit_quality_success`。
   - 私有字段穿透：如 `facade_service._index_service._get_dataset`（`main_window.py:1675`）、`viewport._camera`、`viewport._timer` 等多处。
   - service 间环状引用靠 setter 解（`main_window.py:438-449`），构造顺序敏感，"构造函数注入"的描述不完全成立。

2. **"algorithms/ 纯函数层，零外部依赖"**：import 层面成立（无反向依赖），但整个 `photo_pointcloud_matching/` 子包在桌面端是死代码（服务对象是 Web 端）。

3. **"Alembic 迁移"**：未实施（见一.2）。

4. **检测复核页两头对不上**：实现里占四页之一但仍是占位（`InspectionReviewService` 14 行空壳 + 占位画布），README 则完全没提这个功能的存在。

5. **依赖方向**：import 层面无违规（algorithms/models/db/config/utils 均未 import 上层），这条约定守住了，值得在文档中保留。

---

## 四、其他观察到的问题（与文档无关，供参考）

- 死代码仍在：`db/engine.py`、`db/crud.py`、`services/project_service.py`、`services/project_restore_service.py`、`algorithms/facade/quality.py`（旧版 v1）均无调用方；`processing_runs` / `heatmaps` / `reports` 三张表无人写入（注：`reports` 表是否已被新 PDF 渲染链路使用需再确认）。
- `ui/main_window.py` 已超 3000 行，上帝类问题加剧。
- `utils/convert_fls2ply.py` 硬编码 `D:\ElevationDetect\...` 外部 exe 路径。

---

## 五、修改建议（待审核后执行）

1. **重写目录树**：对齐 `facadeDetection/` 实际结构（含 `runtime/`、`dal/`、各子包），删除所有"（示意）"。
2. **拆分双份文档**：README = 项目简介 + 技术栈 + 快速上手（Python 3.12、`.venv`、`cd facadeDetection && python main.py`、FLS 外部依赖声明）；架构细节全部收敛到 `docs/ARCHITECTURE.md`，README 链接。
3. **ARCHITECTURE.md 补写**：存储设计（双库 + pcfd 索引 + Storage 解析顺序）、窗口嵌入方案与平台限制、dal/runtime 层职责、四页工作台、配准/质量/报告三条主链路。
4. **声明 Web 端定位**：`app.py` / `backend/` / `templates/` 标注"保留不维护"。
5. **模块分工表更新到子包级**，负责人按现状重排。
6. （可选）在 ARCHITECTURE.md 中记录已知架构债：上帝类、setter 解环、service 弹窗、UI 直连 DAL，作为后续重构清单。
