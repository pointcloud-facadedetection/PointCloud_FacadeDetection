"""Application runtime lifecycle ownership.

This module owns task pools and cancellation boundaries.  UI code may use the
runtime coordinator, but does not need to know how pools are constructed.
"""

from PySide6.QtCore import QObject, QThreadPool


class RuntimeLifecycle(QObject):
    """Own project-scoped pools and release them in one deterministic place."""

    def __init__(self, parent=None, *, max_thread_count=1):
        super().__init__(parent)
        self._pools = {}
        self._max_thread_count = max(1, int(max_thread_count))

    def pool(self, name):
        key = str(name)
        pool = self._pools.get(key)
        if pool is None:
            pool = QThreadPool(self)
            pool.setMaxThreadCount(self._max_thread_count)
            self._pools[key] = pool
        return pool

    def clear(self):
        for pool in self._pools.values():
            pool.clear()

    def stop(self, timeout=100):
        self.clear()
        for pool in self._pools.values():
            pool.waitForDone(int(timeout))

    def dispose(self, timeout=100):
        self.stop(timeout)
        self._pools.clear()
