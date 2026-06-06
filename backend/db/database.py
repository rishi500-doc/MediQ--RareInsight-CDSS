"""
PostgreSQL async connection pool using asyncpg.
Provides a shared pool used by all Digital Twin DB operations.
"""
import os
import asyncpg
from dotenv import load_dotenv
from backend.utils.common import get_logger

load_dotenv()

logger = get_logger("CDSS.DB")

_pool: asyncpg.Pool | None = None


async def get_db_pool() -> asyncpg.Pool:
    """
    Return the shared asyncpg connection pool.
    Creates the pool on first call (lazy initialisation).
    """
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set in .env. "
                "Add: DATABASE_URL=postgresql://user:pass@localhost:5432/prohealth"
            )
        _pool = await asyncpg.create_pool(
            dsn=database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("PostgreSQL connection pool created.")
    return _pool


async def close_db_pool():
    """Gracefully close the connection pool on app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed.")


async def is_db_available() -> bool:
    """Health check — returns True if DB is reachable."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")
        return False
