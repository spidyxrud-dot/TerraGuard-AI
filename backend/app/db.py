"""Database engine and session management (Phase 6).

The engine is created lazily from ``app.config.Settings.database_url`` so that no
credentials are hard-coded and tests can point at an isolated database via the
``DATABASE_URL`` environment variable.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


@lru_cache(maxsize=1)
def get_engine(database_url: str | None = None) -> Engine:
    """Create (and cache) the SQLAlchemy engine for the given URL.

    A bounded ``connect_timeout`` is enforced so unreachable PostgreSQL fails fast
    instead of stalling API requests on filtered ports.
    """
    url = database_url or settings.database_url
    if "connect_timeout" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}connect_timeout=3"
    return create_engine(url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def database_available(database_url: str | None = None) -> bool:
    """Cheap connectivity probe used by API endpoints to degrade gracefully."""
    try:
        with get_engine(database_url).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def postgis_available(database_url: str | None = None) -> bool | None:
    """True/False when a database answered, None when unreachable."""
    try:
        with get_engine(database_url).connect() as conn:
            row = conn.execute(
                text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis')")
            ).scalar_one()
        return bool(row)
    except Exception:
        return None
