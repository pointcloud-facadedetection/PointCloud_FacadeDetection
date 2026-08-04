from __future__ import annotations

from typing import Optional, Sequence, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

T = TypeVar("T")


def create(session: Session, obj: T) -> T:
    """
    Add a new object to the session and flush.

    Args:
        session: SQLAlchemy Session
        obj: ORM model instance

    Returns:
        The same instance (with ID populated if applicable)
    """
    session.add(obj)
    session.flush()
    return obj


def get(session: Session, model: Type[T], id_: int) -> Optional[T]:
    """
    Retrieve an object by primary key.

    Args:
        session: SQLAlchemy Session
        model: ORM model class
        id_: Primary key value

    Returns:
        The object if found, else None
    """
    return session.get(model, id_)


def list_all(session: Session, model: Type[T]) -> Sequence[T]:
    """
    Fetch all records of a given model.

    Args:
        session: SQLAlchemy Session
        model: ORM model class

    Returns:
        Sequence of all instances
    """
    return session.execute(select(model)).scalars().all()


def update(session: Session) -> None:
    """
    Flush pending changes to the database.

    This is a convenience wrapper around session.flush().
    Usually you don't need to call it explicitly if you use
    the context manager that auto-commits, but it's useful
    for partial updates within a transaction.

    Args:
        session: SQLAlchemy Session
    """
    session.flush()


def soft_delete(session: Session, obj) -> None:
    """
    Perform a soft-delete if the object has an 'is_deleted' flag,
    otherwise perform a hard delete.

    Args:
        session: SQLAlchemy Session
        obj: ORM model instance
    """
    if hasattr(obj, "is_deleted"):
        setattr(obj, "is_deleted", True)
    else:
        session.delete(obj)