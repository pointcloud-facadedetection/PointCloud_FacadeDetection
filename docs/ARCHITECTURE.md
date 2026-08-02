# PointCloud_FacadeDetection 架构

## 技术栈

| 层面 | 技术 |
|------|------|
| 应用框架 | PySide6 |
| 前端渲染 | Open3D Widget 嵌入 PySide6 |
| 数据存储 | SQLite + SQLAlchemy ORM |

## 3D 渲染控件选型

`Open3D` `visualization.gui` 嵌入 PySide6 `QWidget.createWindowContainer()`。

理由：点云渲染/拾取/交互开箱即用，项目已有依赖。

## 目录结构

```
PointCloud_FacadeDetection/
│
├── facadeDetection/main.py         # VS Code 默认桌面端入口
│
├── config/                         # 【配置】
│   └── settings.py                 # 全局常量（示意）
│
├── ui/                             # 【UI 层】— haolin
│   ├── main_window.py              # 主窗口（示意）
│   ├── widgets/                    # 面板组件（示意）
│   ├── dialogs/                    # 弹窗（示意）
│   └── resources/                  # 资源文件（示意）
│
├── view3d/                         # 【3D 渲染控件】— ruiqi
│   ├── base_viewport.py            # 视口抽象接口（示意）
│   └── open3d_viewport.py          # Open3D 视口实现（示意）
│
├── db/                             # 【数据库基础设施】— haolin
│   ├── connection.py               # 引擎 / 会话工厂（示意）
│   └── migrations/                 # 表结构迁移（示意）
│
├── models/                         # 【ORM 模型】— haolin
│   ├── project.py                  # 项目表（示意）
│   ├── pointcloud.py               # 点云表（示意）
│   ├── analysis.py                 # 分析结果表（示意）
│   └── registration.py             # 配准记录表（示意）
│
├── algorithms/                     # 【算法层】— liying / ruiqi / tiexin
│
├── services/                       # 【业务编排层】— haolin
│   └── ...                         # 流程编排（示意）
│
├── utils/                          # 【工具】
│   └── ...                         # 文件转换等（示意）
```

## 接口层职责

services/ 负责流程编排，代码模式固定为四步：**取数据 → 调算法 → 写结果 → 通知渲染**。

所有 service 类通过构造函数注入 `viewport`、`db`，对外暴露的方法签名即为 UI 层可调用的接口。

> 示例见 `facadeDetection/services/facade_service.py`

## 模块分工

| 模块 | 负责人 | 说明 |
|------|--------|------|
| `ui/` | haolin | PySide6 界面、窗口布局、面板组件、样式 |
| `view3d/` | ruiqi | Open3D 视口嵌入、点云渲染、拾取交互 |
| `db/` | haolin | 数据库连接、会话管理、Alembic 迁移 |
| `models/` | haolin | SQLAlchemy ORM 表定义 |
| `algorithms/` | liying / ruiqi / tiexin | 纯算法实现（检测、分割、配准、质量评估） |
| `services/` | haolin | 业务编排，串联上述模块完成用户操作流程 |
| `config/` | 共享 | 全局常量 |
| `utils/` | 共享 | 文件转换等工具 |

## 依赖方向

```
ui/ + view3d/
    ↓
services/
    ↓
models/     algorithms/     utils/
    ↓
  db/
```

上层可 import 下层，下层严禁 import 上层。`algorithms/` 为纯函数层，零外部依赖。
