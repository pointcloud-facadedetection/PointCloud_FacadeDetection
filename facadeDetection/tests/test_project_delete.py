"""删除项目：只释放被删项目的数据库引擎，且释放后 project.db 可被删除。

旧实现用 lru_cache.cache_clear() 丢弃全部项目引擎而不 dispose，其他项目的
SQLite 连接要等垃圾回收才关闭；之后删除那些项目时 Windows 报 WinError 32。
"""
import gc
import shutil

from sqlalchemy import select

from config.storage import Storage
from db.connection import _PROJECT_ENGINES, dispose_project_engine, project_session
from models import Project


def test_dispose_releases_only_target_project_db(tmp_path, monkeypatch):
    monkeypatch.setattr(Storage, 'project_db_path',
                        lambda pid: tmp_path / pid / 'project.db')
    ids = ('test-delete-a', 'test-delete-b')
    gc.disable()  # 不依赖垃圾回收关闭连接
    try:
        for pid in ids:
            with project_session(pid) as session:
                session.execute(select(Project)).all()

        dispose_project_engine(ids[0])
        assert ids[0] not in _PROJECT_ENGINES
        assert ids[1] in _PROJECT_ENGINES
        shutil.rmtree(tmp_path / ids[0])

        dispose_project_engine(ids[1])
        shutil.rmtree(tmp_path / ids[1])
    finally:
        gc.enable()
        for pid in ids:
            dispose_project_engine(pid)


def test_dispose_unknown_project_does_not_create_db(tmp_path, monkeypatch):
    monkeypatch.setattr(Storage, 'project_db_path',
                        lambda pid: tmp_path / pid / 'project.db')
    dispose_project_engine('test-never-opened')
    assert not (tmp_path / 'test-never-opened').exists()
