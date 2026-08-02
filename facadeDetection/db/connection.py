from contextlib import contextmanager
import os
from functools import lru_cache
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from config.settings import Config  
from models import Base as GlobalBase  

# Global (shared) database at Config.BASE_DIR/data.db (per decision)
DATABASE_URL = f"sqlite:///{Config.BASE_DIR}/data.db"

# Global engine and session
engine_global = create_engine(
    DATABASE_URL,
    future=True,
    echo=False,
    connect_args={
        "check_same_thread": False  # allow access from background threads
    },
)
SessionGlobal = sessionmaker(bind=engine_global, expire_on_commit=False, autoflush=False)


def _apply_sqlite_pragmas(dbapi_con, con_record):
    # Ensure better concurrency and FK integrity
    cursor = dbapi_con.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        # Some SQLite builds may not allow changing journal mode per connection
        pass
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


# Apply PRAGMAs on connect for global DB
event.listen(engine_global, "connect", _apply_sqlite_pragmas)


def init_global_schema():
    """
    Import models and create tables for the global DB.
    Call this once at app start or lazily before first use.
    """
    # Import modules to register models with GlobalBase.metadata
    import models.project  # noqa: F401
    import models.pointcloud  # noqa: F401
    import models.analysis  # noqa: F401
    import models.registration  # noqa: F401
    import models.files  # noqa: F401

    GlobalBase.metadata.create_all(engine_global)


@contextmanager
def global_session():
    """
    Context-managed session for the global DB.
    Commits on success; rolls back on exceptions.
    """
    session = SessionGlobal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Backward-compatible generator for frameworks expecting dependency style
def get_db():
    db = SessionGlobal()
    try:
        yield db
    finally:
        db.close()


# -------- Per-project database support (optional isolation to reduce locks) --------

def get_project_db_dir(project_id: int, directory_path: str | None = None) -> str:
    """
    Resolve per-project DB directory.
    If directory_path is provided (stored on Project.directory_path), use it; otherwise default under UPLOAD_FOLDER.
    """
    base_dir = directory_path or os.path.join(Config.UPLOAD_FOLDER, str(project_id))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_project_db_path(project_id: int, directory_path: str | None = None) -> str:
    """
    Resolve per-project DB file path.
    """
    dir_path = get_project_db_dir(project_id, directory_path)
    return os.path.join(dir_path, f"project_{project_id}.db")


@lru_cache(maxsize=128)
def get_project_engine(project_id: int, directory_path: str | None = None):
    """
    Lazily create and cache a SQLAlchemy engine for a project's dedicated SQLite DB.
    """
    db_path = get_project_db_path(project_id, directory_path)
    url = f"sqlite:///{db_path}"
    engine = create_engine(
        url,
        future=True,
        echo=False,
        connect_args={
            "check_same_thread": False
        },
    )
    # Apply PRAGMAs for the project DB as well
    event.listen(engine, "connect", _apply_sqlite_pragmas)
    return engine

def ensure_project_schema(project_id: int, directory_path: str | None = None):
    """
    Ensure the per-project DB schema exists by importing project-scoped models and creating tables.
    """
    engine = get_project_engine(project_id, directory_path)
    # Import project-scoped models
    from models.project_models import ProjectBase  # type: ignore
    # import models.project_models.facade  
    # import models.project_models.quality  
    ProjectBase.metadata.create_all(engine)
    return engine

@contextmanager
def project_session(project_id: int, directory_path: str | None = None):
    """
    Context-managed session for a project's dedicated DB.
    Ensures schema, commits on success; rolls back on failure.
    """
    engine = ensure_project_schema(project_id, directory_path)
    SessionProject = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    session = SessionProject()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# ------------- Backward-compatible aliases for existing code -------------
# Some modules expect 'engine' and 'SessionLocal'; expose aliases:
engine = engine_global
SessionLocal = SessionGlobal
