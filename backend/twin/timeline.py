"""
Timeline Engine — Records and retrieves longitudinal clinical events.

Every clinical event (symptom onset, lab result, diagnosis, treatment) is
written to twin_timeline_events so the patient's health journey can be
visualised and queried chronologically.
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta, date
from typing import Optional, List

from backend.db.database import get_db_pool
from backend.utils.common import get_logger
from backend.models.twin_schemas import TimelineEventCreate

logger = get_logger("CDSS.Twin.Timeline")


class TimelineEngine:
    """Manages the longitudinal event log for all patient twins."""

    # ── Write ─────────────────────────────────────────────────────────────────

    async def add_event(self, twin_id: str, event: TimelineEventCreate) -> dict:
        """
        Record a new clinical event on the patient timeline.
        Triggers a twin embedding refresh if the event includes new HPO terms.
        """
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO twin_timeline_events
                    (twin_id, event_date, event_type, title, description,
                     severity, hpo_terms, source_id, source_table, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7,
                        $8::uuid, $9, $10::jsonb)
                RETURNING *
            """,
                twin_id,
                event.event_date,
                event.event_type.value,
                event.title,
                event.description,
                event.severity.value,
                event.hpo_terms,
                event.source_id,
                event.source_table,
                json.dumps(event.metadata),
            )

            # Update twin's last_updated timestamp
            await conn.execute("""
                UPDATE patient_twins
                SET last_updated = NOW(), version = version + 1
                WHERE twin_id = $1::uuid
            """, twin_id)

        result = dict(row)
        logger.info(
            f"Timeline event added: [{event.event_type.value}] '{event.title}' "
            f"for twin {twin_id}"
        )

        # Non-blocking embedding refresh if new HPO terms added
        if event.hpo_terms:
            await self._refresh_twin_embedding(twin_id)

        return _serialise(result)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_timeline(
        self,
        twin_id: str,
        days_back: int = 365,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> dict:
        """
        Return the chronological timeline for a patient.

        Args:
            twin_id:    Patient twin UUID
            days_back:  How many days back to query (default 1 year)
            event_type: Filter by specific event type (optional)
            severity:   Filter by severity level (optional)
        """
        pool = await get_db_pool()
        since = (datetime.utcnow() - timedelta(days=days_back)).date()

        conditions = ["twin_id = $1::uuid", "event_date >= $2"]
        params: list = [twin_id, since]
        idx = 3

        if event_type:
            conditions.append(f"event_type = ${idx}")
            params.append(event_type)
            idx += 1

        if severity:
            conditions.append(f"severity = ${idx}")
            params.append(severity)
            idx += 1

        where_clause = " AND ".join(conditions)

        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM twin_timeline_events
                WHERE {where_clause}
                ORDER BY event_date DESC, event_timestamp DESC
            """, *params)

        events = [_serialise(dict(r)) for r in rows]
        return {
            "twin_id" : twin_id,
            "events"  : events,
            "total"   : len(events),
            "since"   : since.isoformat(),
        }

    async def get_lab_trend(self, twin_id: str, test_name: str) -> dict:
        """
        Return time-series values for a specific lab test.
        Includes trend direction (increasing / decreasing / stable).
        """
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT test_date, value_numeric, interpretation, unit, reference_low, reference_high
                FROM twin_lab_results
                WHERE twin_id = $1::uuid AND test_name ILIKE $2
                ORDER BY test_date ASC
            """, twin_id, f"%{test_name}%")

        data_points = [_serialise(dict(r)) for r in rows]

        # Compute trend direction
        trend = "stable"
        if len(data_points) >= 2:
            first = data_points[0].get("value_numeric")
            last  = data_points[-1].get("value_numeric")
            if first and last and first != 0:
                pct_change = (last - first) / abs(first)
                if pct_change > 0.05:
                    trend = "increasing"
                elif pct_change < -0.05:
                    trend = "decreasing"

        return {
            "test_name"    : test_name,
            "twin_id"      : twin_id,
            "trend"        : trend,
            "data_points"  : data_points,
            "count"        : len(data_points),
            "latest_value" : data_points[-1] if data_points else None,
        }

    async def get_milestone_summary(self, twin_id: str) -> dict:
        """
        Return key diagnostic milestones and compute diagnostic delay.
        Diagnostic delay = days between first symptom and first confirmed diagnosis.
        """
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT event_date, event_type, title, severity, hpo_terms
                FROM twin_timeline_events
                WHERE twin_id = $1::uuid
                  AND event_type IN (
                      'symptom_onset', 'diagnosis', 'genetic_result',
                      'treatment_start', 'hospitalization'
                  )
                ORDER BY event_date ASC
            """, twin_id)

        events = [_serialise(dict(r)) for r in rows]

        # Compute diagnostic delay
        first_symptom = next(
            (e for e in events if e["event_type"] == "symptom_onset"), None
        )
        first_diagnosis = next(
            (e for e in events if e["event_type"] == "diagnosis"), None
        )

        delay_days = None
        if first_symptom and first_diagnosis:
            from datetime import date as dt_date
            d1 = date.fromisoformat(first_symptom["event_date"])
            d2 = date.fromisoformat(first_diagnosis["event_date"])
            delay_days = (d2 - d1).days

        return {
            "twin_id"              : twin_id,
            "milestones"           : events,
            "first_symptom_date"   : first_symptom["event_date"] if first_symptom else None,
            "first_diagnosis_date" : first_diagnosis["event_date"] if first_diagnosis else None,
            "diagnostic_delay_days": delay_days,
        }

    async def get_recent_alerts(self, twin_id: str, limit: int = 5) -> list:
        """Return the most recent high/critical severity events."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM twin_timeline_events
                WHERE twin_id = $1::uuid
                  AND severity IN ('high', 'critical')
                ORDER BY event_timestamp DESC
                LIMIT $2
            """, twin_id, limit)
        return [_serialise(dict(r)) for r in rows]

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _refresh_twin_embedding(self, twin_id: str):
        """Non-blocking embedding refresh when phenotype changes."""
        try:
            from backend.twin.similarity import upsert_twin_embedding
            from backend.twin.builder import TwinBuilder
            builder = TwinBuilder()
            twin_data = await builder.get_twin(twin_id)
            if twin_data:
                upsert_twin_embedding(twin_id, twin_data)
                logger.info(f"ChromaDB embedding refreshed for twin {twin_id}")
        except Exception as e:
            logger.warning(f"Embedding refresh failed for {twin_id}: {e}")


# ─── Serialisation Helper ──────────────────────────────────────────────────────

def _serialise(obj):
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialise(i) for i in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif type(obj).__name__ in ("UUID",):
        return str(obj)
    return obj
