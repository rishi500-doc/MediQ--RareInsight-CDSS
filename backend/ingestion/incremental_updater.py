"""
Rare Disease CDSS - Incremental Updater
Tracks ingestion state.
"""
import json
import os
import time

class IncrementalUpdater:
    def __init__(self, state_file="ingestion_state.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f)

    def is_indexed(self, doc_id: str) -> bool:
        return doc_id in self.state

    def mark_indexed(self, doc_id: str):
        self.state[doc_id] = time.time()
        self.save_state()
