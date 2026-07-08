import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DEFAULT_VOXEL_SIZE = 0.05

    QUALITY_GRID_SIZE = 20.0
    RULER_SIZE = 2.0
    RULER_STEP = 0.05

    FACADE_MAX_ITERATIONS = 40
    FACADE_NORMAL_ANGLE_DEG = 18.0
    FACADE_MIN_POINTS_RATIO = 0.003

    SEGMENT_COLORS = [
        [1.00, 0.20, 0.20],
        [1.00, 0.85, 0.10],
        [0.10, 0.45, 1.00],
        [0.20, 0.85, 0.35],
        [0.85, 0.25, 1.00],
        [1.00, 0.50, 0.10],
        [0.10, 0.90, 0.90],
        [0.95, 0.95, 0.95],
    ]

    FACADE_TYPE_COLORS = {
        'vertical_facade': [0.95, 0.68, 0.18],
        'horizontal': [0.25, 0.65, 0.95],
        'inclined': [0.55, 0.45, 0.9],
    }

    FACADE_DIM_COLORS = {
        'vertical_facade': [0.55, 0.38, 0.12],
        'horizontal': [0.12, 0.34, 0.5],
        'inclined': [0.28, 0.22, 0.45],
    }

    HIGHLIGHT_COLOR = [0.2, 0.8, 0.2]
