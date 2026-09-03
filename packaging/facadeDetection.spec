# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import Tree, collect_all, collect_data_files, collect_submodules


project_root = Path(__file__).resolve().parents[1]
app_root = project_root / 'facadeDetection'

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# 基础项目资源
datas += collect_data_files('facadeDetection', includes=['utils/pca-code.json'])

# 尽量完整收集关键第三方包，减少手工漏项
for pkg in ['PySide6', 'open3d', 'qtwebview2', 'numpy', 'opencv_python', 'sqlalchemy', 'pypinyin']:
    try:
        pkg_datas, pkg_bins, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_bins
        hiddenimports += pkg_hidden
    except Exception:
        pass

# facadeDetection 自身的动态导入/包边界
hiddenimports += collect_submodules('facadeDetection')

# 运行时 hook，保证 pybind DLL/ pyd 能从 exe 同级目录加载
runtime_hooks = [str(project_root / 'packaging' / 'runtime_hook_fls_converter.py')]

# 外部 pybind11 包不应被 Python 分析器重命名或压入 archive；原样放到
# dist/facadeDetection/FlsConverter_package/FlsConverter，供 DLL 搜索路径使用。
converter_root = project_root / 'FlsConverter_package'
if converter_root.is_dir():
    datas += Tree(str(converter_root), prefix='FlsConverter_package')

# main.py 内部使用 from ui / from db / from services 等顶层导入；同时保留
# 项目根目录，保证 facadeDetection 包本身及其资源可以被分析。
pathex = [str(app_root), str(project_root)]

a = Analysis(
    [str(app_root / 'main.py')],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=['matplotlib', 'scipy', 'torch', 'tensorflow'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='facadeDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='facadeDetection',
)
