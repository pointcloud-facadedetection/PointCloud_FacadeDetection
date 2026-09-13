"""【创建/编辑项目】表单与通用对话框的统一样式。

单独成文件的原因：``APPLICATION_STYLE_SHEET`` 已经很长，把表单相关的
样式追加在一处便于维护，也让 ``apply_application_theme`` 只需做一次
字符串拼接即可完成挂载。

覆盖范围（与图 3 的改造目标一致）：
- 表单标题层级（标题 + 副标题 + 分区标题）
- 必填项星号、错误态输入框（``formState="error"``）
- 字段下方的校验/提示文案（``fieldHint`` / ``fieldHintError``）
- 文件导入列表的行样式与空态提示
- 底部操作条的按钮排布

所有规则都只使用对象名或动态属性选择器，不依赖具体类的布局，因此不会
影响既有业务逻辑与控件连接。
"""

FORM_STYLE_SHEET = """
/* ------------------------------------------------------------------
   【创建/编辑项目】表单容器
   ------------------------------------------------------------------ */
QDialog#projectCreateDialog {
    background-color: #F5F7FB;
}

QLabel#formDialogTitle {
    color: #0F172A;
    font-size: 15pt;
    font-weight: 700;
}

QLabel#formDialogSubtitle {
    color: #64748B;
    font-size: 9pt;
}

QTabWidget#formTabs::pane {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    top: -1px;
}

QTabWidget#formTabs QTabBar::tab {
    background-color: #EEF2F7;
    color: #475569;
    border: 1px solid #E2E8F0;
    border-bottom: none;
    padding: 7px 18px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 10pt;
}

QTabWidget#formTabs QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #1E40AF;
    font-weight: 600;
}

QTabWidget#formTabs QTabBar::tab:hover:!selected {
    background-color: #E4EBF5;
    color: #334155;
}

/* 表单页统一留白，避免各页缩进不一致 */
QWidget#formTabPage {
    background-color: #FFFFFF;
}

/* ------------------------------------------------------------------
   分区标题与提示文案
   ------------------------------------------------------------------ */
QLabel[uiRole="formSection"] {
    color: #1E3A5F;
    font-size: 10.5pt;
    font-weight: 600;
    padding: 2px 0 4px 0;
}

QLabel[uiRole="formHint"] {
    color: #64748B;
    font-size: 8.5pt;
}

QLabel#standardHintChip {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 4px;
    color: #1E40AF;
    font-size: 9pt;
    padding: 4px 8px;
}

/* 字段级校验提示：默认隐藏，出错时才显示并转为红色 */
QLabel[uiRole="fieldHint"] {
    color: #64748B;
    font-size: 8pt;
}

QLabel[uiRole="fieldHint"][fieldState="error"] {
    color: #DC2626;
    font-weight: 600;
}

/* 必填星号 */
QLabel[uiRole="requiredMark"] {
    color: #DC2626;
    font-weight: 700;
}

/* 错误态输入框：边框转红，聚焦时加深 */
QLineEdit[fieldState="error"],
QComboBox[fieldState="error"],
QDoubleSpinBox[fieldState="error"],
QSpinBox[fieldState="error"],
QTextEdit[fieldState="error"] {
    border: 1px solid #DC2626;
    background-color: #FEF2F2;
}

QLineEdit[fieldState="error"]:focus,
QComboBox[fieldState="error"]:focus,
QDoubleSpinBox[fieldState="error"]:focus,
QSpinBox[fieldState="error"]:focus,
QTextEdit[fieldState="error"]:focus {
    border: 1px solid #B91C1C;
    background-color: #FFFFFF;
}

/* 表单内的输入控件统一高度与圆角，排版更整齐 */
QWidget#formTabPage QLineEdit,
QWidget#formTabPage QComboBox,
QWidget#formTabPage QDoubleSpinBox,
QWidget#formTabPage QSpinBox,
QWidget#formTabPage QDateEdit {
    min-height: 26px;
    border-radius: 4px;
}

/* ------------------------------------------------------------------
   检测参数页：参数分组卡片
   ------------------------------------------------------------------ */
QGroupBox#paramGroup {
    background-color: #FBFCFE;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 10px 8px 10px;
    font-size: 10pt;
    font-weight: 600;
    color: #1E3A5F;
}

QGroupBox#paramGroup::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: #FBFCFE;
}

/* ------------------------------------------------------------------
   文件导入页：资源列表
   ------------------------------------------------------------------ */
QGroupBox#importGroup {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px 8px 6px 8px;
    font-size: 10pt;
    font-weight: 600;
    color: #1E3A5F;
}

QGroupBox#importGroup::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}

QListWidget[uiRole="resourceList"] {
    background-color: #FBFCFE;
    border: 1px dashed #CBD5E1;
    border-radius: 6px;
    padding: 3px;
}

QListWidget[uiRole="resourceList"]:focus {
    border: 1px dashed #93B4F5;
}

/* 资源条目的"删除"按钮：默认低调，悬停才转红，减少视觉噪声 */
QPushButton[uiRole="resourceRemove"] {
    background-color: transparent;
    border: 1px solid transparent;
    color: #64748B;
    min-width: 44px;
    padding: 2px 8px;
    font-size: 9pt;
}

QPushButton[uiRole="resourceRemove"]:hover {
    background-color: #FEF2F2;
    border: 1px solid #FECACA;
    color: #DC2626;
}

QLabel[uiRole="resourcePath"] {
    color: #334155;
    font-size: 9pt;
}

QLabel[uiRole="importEmptyHint"] {
    color: #94A3B8;
    font-size: 8.5pt;
}

/* ------------------------------------------------------------------
   底部操作条
   ------------------------------------------------------------------ */
QWidget#formFooter {
    background-color: #F5F7FB;
    border-top: 1px solid #E2E8F0;
}

QLabel[uiRole="formFooterHint"] {
    color: #94A3B8;
    font-size: 8.5pt;
}
"""
