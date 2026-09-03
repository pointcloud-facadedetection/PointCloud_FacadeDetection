"""UI-facing rendering boundary.

Qt widgets use this small adapter instead of reaching into the viewport
implementation. Rendering policy remains in :mod:`viewport_render_service`.
"""


class ViewportRenderFacade:
    def __init__(self, render_service):
        self._render_service = render_service

    def widget(self):
        return self._render_service.get_widget()

    def select_facade(self, cloud_name, facade_id):
        return self._render_service.select_facade(cloud_name, facade_id)

    def highlight_facades(self, cloud_name, facades):
        return self._render_service.highlight_facades(cloud_name, facades)

    def quality_reports(self, cloud_name, facades, index_service,
                        heatmap_mode='flatness'):
        return self._render_service.render_quality_reports(
            cloud_name, facades, index_service=index_service,
            heatmap_mode=heatmap_mode)

    def compatible_quality_reports(self, cloud_name, facades, index_service):
        return self._render_service.compatible_quality_reports(
            cloud_name, facades, index_service=index_service)

    def apply_quality_colors(self, cloud_name, quality, index_service):
        return self._render_service.apply_quality_colors(
            cloud_name, quality, index_service=index_service)

    def restore_highlight(self, cloud_name, facades):
        return self._render_service.restore_highlight(cloud_name, facades)

    def facade_color(self, facade, order=0):
        return self._render_service.facade_color_for(facade, order)
