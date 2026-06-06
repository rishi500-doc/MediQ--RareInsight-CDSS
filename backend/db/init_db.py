"""
Database initialisation utility.
Reads and executes all SQL migration files in order.
Called on FastAPI startup via the lifespan event.
"""
import os
import asyncio
from pathlib import Path
from backend.db.database import get_db_pool
from backend.utils.common import get_logger

logger = get_logger("CDSS.DB.Init")

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def run_migrations():
    """
    Execute all *.sql migration files in alphabetical (numerical) order.
    Uses IF NOT EXISTS statements so re-runs are safe.
    """
    if not MIGRATIONS_DIR.exists():
        logger.warning(f"Migrations directory not found: {MIGRATIONS_DIR}")
        return

    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        logger.info("No migration files found.")
        return

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        for sql_file in sql_files:
            logger.info(f"Running migration: {sql_file.name}")
            sql = sql_file.read_text(encoding="utf-8")
            try:
                await conn.execute(sql)
                logger.info(f"✓ Migration applied: {sql_file.name}")
            except Exception as e:
                logger.error(f"✗ Migration failed [{sql_file.name}]: {e}")
                raise


if __name__ == "__main__":
    # Allow running directly: python -m backend.db.init_db
    asyncio.run(run_migrations())
    print("All migrations complete.")
