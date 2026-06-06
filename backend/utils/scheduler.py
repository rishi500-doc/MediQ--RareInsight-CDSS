"""
Rare Disease CDSS - Maintenance Scheduler
Automates weekly updates for the HPO ontology and clinical disease database.
"""

import time
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from backend.ingestion.ingestion_manager import IngestionManager
from backend.hpo.hpo_updater import HPOUpdater
from backend.utils.common import get_logger
import asyncio

logger = get_logger("CDSS.Scheduler")

def run_maintenance_cycle():
    """Runs the full maintenance cycle: HPO update followed by data ingestion."""
    logger.info("--- STARTING WEEKLY MAINTENANCE CYCLE ---")
    
    # 1. Update HPO Ontology
    try:
        logger.info("Step 1: Updating HPO Ontology...")
        updater = HPOUpdater()
        updater.update()
    except Exception as e:
        logger.error(f"HPO Update failed during maintenance: {e}")

    # 2. Update Clinical Disease Database (Fetcher)
    try:
        logger.info("Step 2: Updating Clinical Disease Database...")
        manager = IngestionManager()
        asyncio.run(manager.ingest_pipeline())
    except Exception as e:
        logger.error(f"Clinical data ingestion failed during maintenance: {e}")
        
    logger.info("--- MAINTENANCE CYCLE COMPLETE ---")

def start_scheduler():
    """Configures and starts the weekly scheduler."""
    scheduler = BlockingScheduler()
    
    # Schedule the maintenance cycle to run every Sunday at 02:00 AM
    scheduler.add_job(
        run_maintenance_cycle, 
        'cron', 
        day_of_week='sun', 
        hour=2, 
        minute=0,
        id='weekly_maintenance',
        replace_existing=True
    )
    
    logger.info("Scheduler started: Maintenance cycle scheduled for every Sunday at 02:00 AM.")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")

if __name__ == "__main__":
    # Option: Run once immediately if needed, then start schedule
    # run_maintenance_cycle()
    start_scheduler()
