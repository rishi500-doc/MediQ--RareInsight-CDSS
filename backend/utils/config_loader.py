"""
Rare Disease CDSS - Configuration Loader
Handles safe loading, validation, and caching of clinical JSON configurations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from backend.utils.common import get_logger

logger = get_logger("CDSS.ConfigLoader")

class ConfigLoader:
    _cache: Dict[str, Any] = {}

    @classmethod
    def load_json(cls, filename: str) -> Dict[str, Any]:
        """Loads a JSON config file from the data directory with caching."""
        if filename in cls._cache:
            return cls._cache[filename]

        # Resolve path relative to workspace root
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(BASE_DIR, "data", filename)

        if not os.path.exists(file_path):
            logger.warning(f"Config file not found: {file_path}. Returning empty dictionary.")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cls._cache[filename] = data
                logger.info(f"Successfully loaded configuration: {filename}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {filename}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}

    @classmethod
    def get_hallmarks(cls) -> Dict[str, Any]:
        """Returns the Hallmark configuration."""
        return cls.load_json("hallmarks.json")

    @classmethod
    def get_synonyms(cls) -> Dict[str, Any]:
        """Returns the Clinical Synonym configuration."""
        return cls.load_json("clinical_synonyms.json")
