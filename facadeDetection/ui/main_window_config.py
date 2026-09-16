PAGE_DEFINITIONS = (
    ('项目概览', 'project_overview'),
    ('项目操作', 'project_operation'),
    ('检测复核', 'inspection_review'),
    ('报告预览/导出', 'report_export'),
)


PAGE_BUTTON_NAMES = {
    'project_overview': 'btn_overview',
    'project_operation': 'btn_operation',
    'inspection_review': 'btn_inspection_review',
    'report_export': 'btn_report_export',
}

PAGE_HEADER_ACTIONS = {
    'project_overview': (
        ('打开项目文件夹', 'btn_open_project'),
        ('新建项目', 'btn_new_project'),
    ),
    'project_operation': (
        ('点云去噪', 'btn_denoise'),
        ('点云配准', 'btn_registration'),
        ('框选检测区域', 'btn_select_detection_area'),
        ('立面检测', 'btn_facade_detection'),
        ('质量检测', 'btn_quality_inspection'),
        ('框选分割', 'btn_box_segmentation'),
        ('计算细节', 'btn_calculate_detail'),
        ('二维-三维对齐', 'btn_align_2d_3d'),
    ),
    'inspection_review': (),
    'report_export': (
        # 【打开 PDF】已按业务要求整体重构为【导出模型】：
        # 选择项目站点 → 源点云 voxel=0.2 下采样 → 导出 PLY。
        # 对象名沿用 btn_open_report_pdf，避免影响页面装配顺序与测试断言；
        # 控件文案、工具提示与连接的回调均已替换为导出模型语义。
        ('导出模型', 'btn_open_report_pdf'),
        ('导出质量报告', 'btn_export_quality_report'),
    ),
}

APPLICATION_TITLE = '点云外立面智能检测平台'

# 每个页面只突出一个主要操作，避免顶部十余个按钮全部使用主色。
PRIMARY_HEADER_ACTIONS = {
    'btn_facade_detection',
    'btn_open_report_pdf',
}

# Corporate Clean 强调清晰的功能分组。按钮名称和业务连接保持不变，
# 这里只定义它们在顶部命令栏中的视觉归属。
PAGE_HEADER_GROUPS = {
    'project_overview': (
        (
            '项目管理',
            (
                'btn_open_project',
                'btn_new_project',
            ),
        ),
    ),
    'project_operation': (
        ('点云处理', ('btn_denoise', 'btn_registration')),
        (
            '区域处理',
            (
                'btn_select_detection_area',
                'btn_box_segmentation',
            ),
        ),
        (
            '立面分析',
            (
                'btn_facade_detection',
                'btn_quality_inspection',
            ),
        ),
        (
            '结果关联',
            (
                'btn_calculate_detail',
                'btn_align_2d_3d',
            ),
        ),
    ),
    'inspection_review': (),
    'report_export': (
        # 质量报告导出是 ruiqi_dev 新增能力，必须随当前报告页命令栏一起创建，
        # 后续按钮连接才能安全取得对应控件。
        ('', ('btn_open_report_pdf', 'btn_export_quality_report')),
    ),
}
