# 立面算法已迁入 algorithms.facade 子包，服务层同步使用最新入口。
from algorithms.facade.facade_detection import detect_facades


class FacadeService:

    def __init__(self, viewport, db):
        self._viewport = viewport
        self._db = db

    def detect(self, cloud_name: str, max_iterations: int = 40) -> list[dict]:
        # 1. 从 3D 视口取数据
        pcd = self._viewport.get_cloud(cloud_name)

        # 2. 调用算法层
        results = detect_facades(pcd, max_iterations=max_iterations)

        # 3. 结果写入数据库
        from models.analysis import Analysis
        for r in results:
            self._db.add(Analysis(result_type='facade', result_data=r))
        self._db.commit()

        # 4. 通知 3D 视口更新渲染
        self._viewport.highlight_facades(results)

        return results
