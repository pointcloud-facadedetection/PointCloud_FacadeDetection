from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from datetime import datetime
from typing import Generator
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config.storage import Storage
from models import Base 


def _apply_sqlite_pragmas(dbapi_con, con_record):
    cur = dbapi_con.cursor()
    try:
        cur.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.close()


# -------------------- Global index database (lightweight) --------------------

IndexBase = declarative_base()


class IndexProject(IndexBase):
    __tablename__ = "projects"
    id = __import__("sqlalchemy").Column(__import__("sqlalchemy").Integer, primary_key=True)
    project_uuid = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, unique=True, index=True, nullable=False)
    name = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, nullable=False)
    root_dir = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, nullable=False)
    thumbnail_path = __import__("sqlalchemy").Column(__import__("sqlalchemy").String, nullable=True)
    created_at = __import__("sqlalchemy").Column(__import__("sqlalchemy").DateTime, default=datetime.now, nullable=False)
    updated_at = __import__("sqlalchemy").Column(__import__("sqlalchemy").DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)


Storage.ensure_base_dirs()
_index_engine = create_engine(
    f"sqlite:///{Storage.INDEX_DB_FILE}", future=True, echo=False, connect_args={"check_same_thread": False}
)
event.listen(_index_engine, "connect", _apply_sqlite_pragmas)
IndexSessionFactory = sessionmaker(bind=_index_engine, expire_on_commit=False, autoflush=False, future=True)


def init_index_db() -> None:
    IndexBase.metadata.create_all(_index_engine)


@contextmanager
def index_session() -> Generator[Session, None, None]:
    s: Session = IndexSessionFactory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def upsert_index_project(project_uuid: str, name: str, root_dir: str, thumbnail_path: str | None = None) -> None:
    with index_session() as s:
        row = s.execute(select(IndexProject).where(IndexProject.project_uuid == project_uuid)).scalar_one_or_none()
        if row:
            row.name = name
            row.root_dir = root_dir
            row.thumbnail_path = thumbnail_path
        else:
            s.add(IndexProject(project_uuid=project_uuid, name=name, root_dir=root_dir, thumbnail_path=thumbnail_path))
        s.flush()


def list_index_projects() -> list[IndexProject]:
    with index_session() as s:
        return s.execute(select(IndexProject).order_by(IndexProject.created_at.desc())).scalars().all()


# -------------------- Per-project database --------------------

@lru_cache(maxsize=256)
def get_project_engine(project_uuid: str):
    Storage.ensure_base_dirs()
    db_path = Storage.project_db_path(project_uuid)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}", future=True, echo=False, connect_args={"check_same_thread": False}
    )
    event.listen(engine, "connect", _apply_sqlite_pragmas)
    # Ensure schema exists
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def project_session(project_uuid: str) -> Generator[Session, None, None]:
    engine = get_project_engine(project_uuid)
    SessionProject = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False, future=True)
    s: Session = SessionProject()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()