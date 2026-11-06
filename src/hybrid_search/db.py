"""Connection pool, schema migrations and pgvector helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Arbitrary constant identifying the migration lock, so concurrent deployments migrate one at a time.
MIGRATION_LOCK_ID = 7_420_113


def to_vector_literal(values: Sequence[float]) -> str:
    """Serializes a vector in pgvector's text format, for use with a ::vector cast."""
    return "[" + ",".join(format(float(v), ".8g") for v in values) + "]"


async def open_pool(url: str, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(url, min_size=min_size, max_size=max_size, open=False)
    await pool.open(wait=True, timeout=30)
    return pool


async def migrate(conn: AsyncConnection) -> list[str]:
    """Applies pending SQL migrations in filename order, each in its own transaction.

    The connection must be in autocommit mode so that every migration commits independently.
    """
    if not conn.autocommit:
        raise ValueError("migrate() requires a connection in autocommit mode")
    await conn.execute("select pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
    try:
        await conn.execute(
            "create table if not exists schema_migrations "
            "(version text primary key, applied_at timestamptz not null default now())"
        )
        cursor = await conn.execute("select version from schema_migrations")
        applied = {row[0] for row in await cursor.fetchall()}

        newly_applied: list[str] = []
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.stem in applied:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text(encoding="utf-8"))
                await conn.execute("insert into schema_migrations (version) values (%s)", (path.stem,))
            newly_applied.append(path.stem)
        return newly_applied
    finally:
        await conn.execute("select pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))


class SchemaMissingError(RuntimeError):
    """The tables do not exist yet: migrations have not been applied to this database."""


async def embedding_dimensions(conn: AsyncConnection) -> int:
    """Reads the dimension declared on chunks.embedding, to verify it matches the configured model."""
    cursor = await conn.execute(
        "select atttypmod from pg_attribute where attrelid = to_regclass('chunks') and attname = 'embedding'"
    )
    row = await cursor.fetchone()
    if row is None:
        raise SchemaMissingError("the database schema is missing; run `hybrid-search migrate` first")
    return int(row[0])
