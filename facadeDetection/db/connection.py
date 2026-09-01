from __future__ import annotations

from contextlib import contextmanager
import os
from functools import lru_cache
from datetime import datetime
from sqlalchemy import create_engine, event, select, Column, Integer, String, DateTime, UniqueConstraint, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from config.storage import Storage
from models import Base as GlobalBase


def _apply_sqlite_pragmas(dbapi_con, _con_record):
    cur = dbapi_con.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.close()


# ---------------- Global index DB (projects list) ----------------
Storage.ensure_base_dirs()
INDEX_DB_URL = f"sqlite:///{Storage.INDEX_DB_FILE}"
engine_index = create_engine(
    INDEX_DB_URL,
    future=True,
    echo=False,
    connect_args={"check_same_thread": False},
)
event.listen(engine_index, "connect", _apply_sqlite_pragmas)

IndexBase = declarative_base()


class IndexProject(IndexBase):  # type: ignore
    __tablename__ = "index_projects"
    __table_args__ = (UniqueConstraint("project_uuid", name="uq_index_projects_uuid"),)

    id = Column(Integer, primary_key=True)
    project_uuid = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    root_dir = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)


def init_index_db() -> None:
    IndexBase.metadata.create_all(engine_index)


IndexSession = sessionmaker(bind=engine_index, expire_on_commit=False, autoflush=False, future=True)


@contextmanager
def index_session():
    init_index_db()
    s = IndexSession()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def upsert_index_project(project_uuid: str, name: str, root_dir: str) -> None:
    init_index_db()
    with index_session() as s:
        row = s.execute(select(IndexProject).where(IndexProject.project_uuid == project_uuid)).scalar_one_or_none()
        if row is None:
            row = IndexProject(project_uuid=project_uuid, name=name, root_dir=root_dir)
            s.add(row)
        else:
            row.name = name
            row.root_dir = root_dir


def list_index_projects() -> list[IndexProject]:  # type: ignore
    init_index_db()
    with index_session() as s:
        return s.execute(select(IndexProject)).scalars().all()


# ---------------- Per-project DB (by project UUID) ----------------

@lru_cache(maxsize=256)
def _project_engine(project_uuid: str):
    db_path = Storage.project_db_path(project_uuid)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, future=True, echo=False, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _apply_sqlite_pragmas)
    # Ensure schema (import models)
    GlobalBase.metadata.create_all(engine)
    # create_all 不会迁移现有项目的数据库。
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS quality_inspection_runs (
            id INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL,
            station_id INTEGER,
            facade_id INTEGER,
            facade_key TEXT NOT NULL,
            facade_display_no INTEGER,
            cloud_name TEXT,
            dataset_id TEXT,
            dataset_fingerprint TEXT,
            dataset_revision TEXT,
            standard_id TEXT,
            standard_name TEXT,
            standard_version TEXT,
            interval_size_m REAL,
            parameter_fingerprint TEXT NOT NULL,
            profile_snapshot_json JSON,
            quality_status TEXT NOT NULL DEFAULT 'complete',
            quality_report_json JSON NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )"""))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_quality_runs_project_station ON quality_inspection_runs(project_id, station_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_quality_runs_facade ON quality_inspection_runs(facade_id)"))
        conn.execute(text('DROP TABLE IF EXISTS heatmaps'))
        conn.execute(text('DROP TABLE IF EXISTS processing_runs'))
        columns = {c['name'] for c in inspect(conn).get_columns('facades')}
        additions = {
            'quality_status': 'TEXT NOT NULL DEFAULT \'pending\'',
            'quality_report_json': 'JSON',
            'color_json': 'JSON',
            'dataset_revision': 'TEXT',
            'quality_completed_at': 'DATETIME',
            'station_id': 'INTEGER',
            'dataset_id': 'TEXT',
            'dataset_fingerprint': 'TEXT',
            'display_no': 'INTEGER NOT NULL DEFAULT 1',
            'point_count': 'INTEGER NOT NULL DEFAULT 0',
            'raw_point_count': 'INTEGER NOT NULL DEFAULT 0',
        }
        added_display_no = 'display_no' not in columns
        for name, definition in additions.items():
            if name not in columns:
                conn.execute(text(f'ALTER TABLE facades ADD COLUMN {name} {definition}'))
        if added_display_no:
            conn.execute(text("""
            UPDATE facades
               SET display_no = CAST(substr(label, instr(label, ' ') + 1) AS INTEGER) + 1
             WHERE (display_no IS NULL OR display_no = 1)
               AND label GLOB 'Facade [0-9]*'
            """))
        station_columns = {c['name'] for c in inspect(conn).get_columns('pointcloud_stations')}
        if 'denoise_state_json' not in station_columns:
            conn.execute(text('ALTER TABLE pointcloud_stations ADD COLUMN denoise_state_json JSON'))
    return engine


@contextmanager
def project_session(project_uuid: str):
    engine = _project_engine(project_uuid)
    SessionProject = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False, future=True)
    s = SessionProject()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
