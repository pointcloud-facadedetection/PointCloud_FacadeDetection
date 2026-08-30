"""Canonical measurement mapping shared by viewport, image export and reports."""

HEATMAP_SPECS = {
    'flatness': {
        'title': '平整度热力图',
        'value_key': 'flatness_gap_mm',
        'pass_key': 'flatness_pass',
        'limit_key': 'flatness_limit_mm',
        'file_key': 'flatness',
    },
    'verticality': {
        'title': '垂直度热力图',
        'value_key': 'verticality_deviation_mm',
        'pass_key': 'verticality_pass',
        'limit_key': 'verticality_limit_mm',
        'file_key': 'verticality',
    },
}


def normalize_heatmap_mode(mode) -> str:
    """Only expose production-supported display modes to downstream services."""
    return 'verticality' if str(mode or '').lower() == 'verticality' else 'flatness'


def heatmap_spec(mode):
    return HEATMAP_SPECS[normalize_heatmap_mode(mode)]