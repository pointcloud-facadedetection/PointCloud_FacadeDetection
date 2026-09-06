# 性能链分析报告

> 基线：`charles_optimize`（main_window 拆分后）
> 范围：`facadeDetection/` 桌面端，18 条 `# TODO(...)` 标记的性能/生命周期链
> 方法：静态代码分析，所有结论附 `文件:行号` 证据；实际耗时需运行时 profiling（代码已有 `[PCFD] load.*` 计时日志和 `trace()` 埋点可直接用）
> 状态：**仅分析，未改任何代码**

---

## 1. 点云加载链：调用路径与内存驻留

### 调用路径（全程 GUI 线程同步）

```
_start_load (ui/controllers/project_lifecycle.py:7)
  └─ ProjectOverviewService.upload_files (services/project_overview/project_overview_service.py:290)
       └─ FileService.upload_files (services/file_service.py:43)   ← 逐文件
            ├─ _load_point_cloud → o3d.io.read_point_cloud (:376)
            ├─ read_dist (:87)                          ← .dist 距离文件
            ├─ estimate_elevation_angles (algorithms/geometry.py:21)
            ├─ stratified_proxy_build (algorithms/geometry.py:278)  ← 距离分层代理
            ├─ register_source_asset / register_dataset (services/pointcloud_service.py:61-93)
            │    └─ VoxelCascadeIndex.build (services/pointcloud_index/core.py:148-199)
            └─ ViewportRenderService.show_point_cloud → PointCloudScene.add_cloud (view3d/scene.py:21)
```

GUI 线程同步是**当时的明确设计**（`project_lifecycle.py:10-15` docstring：QRunnable 会触发 Open3D 的 GUI 线程保护）。后台通道已预留但**是死代码**：`create_load_worker`（`project_overview_service.py:41`）和 `PointCloudLoadWorker`（`utils/workers.py:19`）全工程零调用。

### 同时驻留的大对象（N = 原始点数）

| 对象 | 位置 | 量级 |
|---|---|---|
| 原始 pts+cols | `source_assets`（`pointcloud_service.py:81`），颜色经 `np.clip` 再拷一份（`core.py:31`） | N×24B |
| ranges + scan_origins | `read_dist` | N×4B |
| 高度角临时数组 | `pts[:,None,:]-origins[None,:,:]`，**N×M×3 float64**（M=测站数，N=10⁷、M=4 时瞬时 ~960MB） | 瞬时 |
| 代理 CSR | offsets + indices | N×4B |
| proxy 点/颜色 ×3 份 | dataset.index、`scene.point_data`、Open3D 几何缓冲（`scene.py:45` 注释自承复制） | 3×proxy |

另：`normalize_colors` 中 `np.clip` 又整组拷贝一次颜色（`view3d/lod.py:15`）。

### 隐藏的双倍工作

上传成功后 `_start_load` 调 `station_service.show_single`（`project_lifecycle.py:39`）→ `_load_proxy_domain`（`pointcloud_station_service.py:107`）。复用分支要求指纹匹配（`:128`），而指纹只在重建后写入（`:193`）——**首次展示必不命中**，于是 `:137` 重新 `read_point_cloud` + 重跑 `read_dist` + `stratified_proxy_build`。**同一文件在一次上传流程内被完整处理两遍。**

---

## 2. 项目重开链：重复读取清单

每次打开已登记项目（`_prepare_project_activation` → `_start_load('activate')`）必然发生：

| 重复项 | 证据 | 量级 |
|---|---|---|
| 活动站点 PLY 全量重读 | `_dispose_project_runtime` 已清空 datasets（`project_lifecycle.py:182`），复用分支必不命中 → `read_point_cloud`（`pointcloud_station_service.py:137`） | 每站 1 次 |
| .dist 重解析 + 代理重建 | 无去噪状态时 `read_dist` + `stratified_proxy_build` 全量重跑（`:171-186`） | 每站 1 次 |
| **每个 PLY 全文件 SHA-256** | `sync_assets` → `FileRepo.validate_asset` → `_sha256`（`file_repo.py:17-22`，1MB 块全量读） | 每文件 1 遍全读 |
| `sync_assets` 调 2 次 | `open_project` 自身 1 次（`project_overview_service.py:124/141/164`）+ `activate_project` 1 次（`:175`） | ×2 |
| pcfd 索引重复解析 | `load_pcfd_index` 连续调两次，第二次是无效重试（`project_overview_service.py:111-113`） | — |
| 项目列表全量重扫 | `_refresh_project_list` → `list_projects` 对每项目各解析一次 pcfd JSON + 各开一次 DB session（`project_repo.py:201-211`） | 每次刷新 |
| gc + 路径校验 | 每次激活附带全 FileAsset 文件锁扫描 + 逐路径 `exists` stat（`project_repo.py:252-254`） | 每次激活 |

**结论：每次打开项目，每个 PLY 至少被完整读 2 遍（SHA-256 一遍、`read_point_cloud` 一遍），全在 GUI 线程。** 法线/质量结果不重算（已持久化于 `plane_json`/`quality_report_json` + npz），这点是好的。

---

## 3. 类聚合生命周期

| 对象 | 持有者 | 销毁路径 | 评估 |
|---|---|---|---|
| `PointCloudService.datasets/source_assets/decisions` | 内存注册表（`pointcloud_service.py:20-23`） | 项目切换 `clear_runtime`（:40-53）正确置空 | **泄漏缺口**（见下①） |
| `PointCloudScene.point_data` | `view3d/scene.py:15` | `remove_cloud`/`clear`，项目切换经 `viewport.clear()` | 正确 |
| `Open3DAdapter.geometries` | `open3d_adapter.py:15` | `remove_geometry`/`clear`/`destroy`（幂等） | 正确 |
| `VoxelCascadeIndex`（含惰性缓存 `_raw_to_voxel`/`_source_to_proxy`） | `PointCloudDataset.index` | 随 `clear_runtime`，缓存显式置 None | 正确 |
| DB engine | `db/connection.py:92` `lru_cache(256)` | **从不 dispose**，删除项目也不清 | ② 单调增长（上限 256） |
| `RuntimeLifecycle` 任务池 | MainWindow 建 3 池（`main_window.py:204-207`） | `closeEvent` → `runtime.stop(100)` | ③ 见下 |
| `RuntimeTaskScheduler._running/_tokens` | `task_scheduler.py:53-54` | 靠回调 `_finish` 弹出；`cancel_all` 只置标志不移除条目 | 异常未回调则残留（轻量） |
| `PointCloudStationService` 指纹/配准缓存 | `pointcloud_station_service.py:26-27` | `set_project` 清空 | 项目内只增（轻量） |

**泄漏/风险缺口：**

1. **删站点不释放原始点数组**：`release_station_domain` 只删 `key.endswith(":{station_id}:source")` 的 source asset（`pointcloud_service.py:121-122`），而上传链注册的键是 `f"{project_uuid}:source:{stem}"`（`file_service.py:106-107`）——**永不匹配**，删除站点后全量原始点数组残留内存。dataset 键也存在两套命名（`uuid:filename` vs `uuid:station_id`），靠 legacy 迁移兜底。
2. **DB engine 永不 dispose**：进程内单调增长。
3. **关闭竞态**：所有任务池 `maxThreadCount=1`（串行），`closeEvent` 的 `runtime.stop(100)` 只等 100ms 就销毁 viewport——超时后存活任务可能访问已销毁视口（`main_window.py:748` TODO 所指）。另 `_denoise_thread` 只 `requestInterruption()` 不 `wait`（`project_operation_service.py:74-76`）。

---

## 4. 渲染轮询链：隐藏页的纯浪费

- QTimer **33ms** 驱动 `process_events` → `poll`（`open3d_viewport.py:143-146`）。
- `poll` 内 `vis.poll_events()` **固定 30Hz，与页面可见性无关**（`open3d_adapter.py:25,141-160`）；帧提交有节流（空闲 0.2s / 交互 33ms），这部分设计是对的。
- **`set_current_page`（`main_window.py:685-698`）不切页时暂停视口**；现成的 `set_render_enabled`（`open3d_adapter.py:72-75`）**全工程零调用**。Open3D 视口在"项目概览/报告"页时仍在 30Hz 收 GLFW 事件，若有排队的颜色/点更新还会提交不可见帧。
- `_scene_dirty` 是**死标志**：只写不读（`open3d_adapter.py:19,56,158`），实际门控只用 `_render_pending`。
- `_init_ui` 找原生窗口时以 `process_events` 驱动自旋，最长 120s（`open3d_viewport.py:247-251`）。

---

## 5. 热点与读取速度上限

### 每次加载执行一次

- `read_point_cloud`：理论上限是磁盘吞吐 + PLY 解析（ascii/binary 差异大），但**实际被执行 2 次**（见 §1 双倍工作）→ 有效吞吐打 5 折。
- `stratified_proxy_build`（`geometry.py:278-368`）：7 壳层 × 每壳 `np.lexsort` O(n log n)，同样**被执行 2 次**。
- `VoxelCascadeIndex.build`：分层路径 O(n) 廉价；标准路径全量 lexsort O(n log n)。**每次去噪重建**（`pointcloud_service.py:406`）。
- `estimate_elevation_angles` 的 N×M×3 float64 瞬时数组是加载期内存峰值的主凶。

### 每次检测执行（无跨次缓存）

- **法线每次重估**：`facade_detection_service.py:82-92` 检查 `dataset.proxy_normals`，但该字段在 `PointCloudDataset` 上**根本不存在**（`core.py:48-76`），`hasattr` 恒 False → <500k 点时每次都 `estimate_normals`。**这是缓存逻辑的实际 bug，不是性能风格问题。**
- **每次检测两次全量 deepcopy**：`facade_detection.py:417` deepcopy #1；`ensure_normals`（`geometry.py:684-694`）deepcopy #2。法线缺失时还附带 `orient_normals_consistent_tangent_plane(30)`（O(n) 图传播，比估计更贵）。
- n≤100k 时再算一次全量 NN 距离（`facade_detection.py:459`）。

### O(n) 藏在 Python 循环里（大点云下这些是数量级问题）

| 位置 | 问题 |
|---|---|
| `viewport_render_service.py:271-277` | trace 日志块对每个立面**再算一遍**索引映射 + 整云颜色比较——每次显示检测结果付双份（纯日志开销） |
| `viewport_render_service.py:650` | `render_flatness_heatmap` 在立面循环**内**对全部显示点重建 Python dict → O(F×n) |
| `viewport_render_service.py:802-811` | `apply_quality_colors` 逐点 Python 循环 + dict 查询 |
| `core.py:218-222` | `raw_to_voxel_ids` 首次构建逐体素 Python 循环 |
| `pointcloud_service.py:360-364` | 去噪兜底分支逐点 `search_knn_vector_3d` |
| `geometry.py:663-670` | `is_uniform_color` 纯 Python 逐点比较 |
| `viewport_render_service.py:248,336,358,399` | 各种整云 `np.tile` 颜色矩阵（签名任何变化即整云重 tile） |

---

## 6. 优化优先级（按 ROI 排序，供后续专项参考）

**P0 — 低风险高收益（不改算法语义）：**

1. 修 `proxy_normals` 缓存 bug：给 `PointCloudDataset` 加字段或改 `hasattr` 判断 → 检测不再每次重估法线（§5）。
2. 消掉检测链两次 deepcopy 中的一次（`ensure_normals` 的 `inplace=False`）。
3. 删/旁路 `highlight_facades` 的 trace 统计块（或降级为 debug 开关）→ 省一整遍全立面映射（§5）。
4. `set_current_page` 联动 `set_render_enabled` + timer 暂停（方法已存在，接线即可）→ 隐藏页零渲染开销（§4）。
5. SHA-256 校验加 size+mtime 短路；`sync_assets` 合并为一次调用 → 项目打开每文件少读一整遍（§2）。

**P1 — 中风险（动数据流）：**

6. 修上传→`show_single` 的指纹门控，让同一文件一次上传只处理一遍（§1）。
7. `render_flatness_heatmap` 的 dict 移出循环；`apply_quality_colors` 向量化（§5）。
8. 修 `release_station_domain` 键不匹配 → 删站点真正释放内存（§3①）。
9. `estimate_elevation_angles` 分块计算，消掉 N×M×3 瞬时数组（§1）。

**P2 — 结构性（需要设计）：**

10. 加载链后台化：后台解析（read/dist/proxy）+ GUI 线程只提交 Open3D 几何——`create_load_worker`/`PointCloudLoadWorker` 通道已预留，但 Open3D 提交必须留在 GUI 线程，需要拆分"解析"与"提交"两阶段。
11. proxy/颜色三份驻留（dataset/scene/Open3D）收敛为共享 buffer 或惰性提交。
12. DB engine 随项目关闭 dispose；closeEvent 池超时与任务取消协议重做。

---

## 7. 实测手段（已具备）

- 加载链各阶段计时日志：`file_service.py:76-79,153,160`（`[PCFD] load.*`）。
- `trace()` 埋点：`utils/logging_utils.py`，加载/着色链已埋。
- 建议专项时用真实工程文件（等工程师的 FLS/PLY）跑三轮取中位数，对照本文 §1/§2 的双倍工作点验证修复收益。
