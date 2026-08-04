from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base  # type: ignore


def get_session_factory(db_path: str | None = None) -> Callable[[], object]:
    """
    创建一个 SQLite 引擎，并返回一个零参数factory，该factory可生成新的 Session 对象。
    确保metadata已创建。
    """
    if db_path is None:
        root = Path(__file__).resolve().parents[1]
        data_dir = root.parent / 'data'
        os.makedirs(data_dir, exist_ok=True)
        db_path = str(data_dir / 'app.db')

    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri, future=True)

    # Create tables if not exist
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def factory():
        return SessionLocal()

    return factory
