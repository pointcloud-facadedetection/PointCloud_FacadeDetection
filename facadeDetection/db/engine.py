from __future__ import annotations

from typing import Callable

from db.connection import init_index_db, IndexSessionFactory


def get_session_factory() -> Callable[[], object]:
    """
    Back-compat factory that returns sessions bound to the lightweight
    global index database. Prefer using db.connection APIs directly.
    """
    init_index_db()

    def factory():
        return IndexSessionFactory()

    return factory
