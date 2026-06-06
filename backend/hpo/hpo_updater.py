"""
Rare Disease CDSS - HPO Ontology Updater
Downloads the official Human Phenotype Ontology (hp.json) and converts it 
to a high-performance local JSON dictionary for phenotype matching.
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CDSS.HPOUpdater")

class HPOUpdater:
    """Orchestrates the download and parsing of the HPO ontology."""
    
    # Official OBO PURL for the JSON format
    OBO_URL = "http://purl.obolibrary.org/obo/hp.json"
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Default to backend/data/hpo
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(BASE_DIR, "data", "hpo")
            
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.dict_path = os.path.join(self.output_dir, "hpo_dictionary.json")
        self.meta_path = os.path.join(self.output_dir, "metadata.json")

    def should_update(self) -> bool:
        """Return True if the local HPO dictionary is missing or stale."""
        if not os.path.exists(self.dict_path):
            logger.info("HPO dictionary not found locally; update required.")
            return True

        if not os.path.exists(self.meta_path):
            logger.info("HPO metadata not found; update required.")
            return True

        try:
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            timestamp = metadata.get("download_timestamp")
            if timestamp:
                downloaded_at = datetime.fromisoformat(timestamp)
                cache_days = int(os.getenv("HPO_CACHE_AGE_DAYS", "7"))
                if datetime.now() - downloaded_at < timedelta(days=cache_days):
                    logger.info(f"Local HPO data is fresh (cached for {cache_days} days); skipping download.")
                    return False
                logger.info("Local HPO data is stale; update required.")
                return True
        except Exception as e:
            logger.warning(f"Could not read HPO metadata; update required: {e}")
            return True

        return True

    def download_ontology(self) -> str:
        """Downloads the hp.json file to a temporary location."""
        logger.info(f"Downloading HPO ontology from {self.OBO_URL}...")
        temp_path = os.path.join(self.output_dir, "hp_raw.json")
        
        try:
            response = requests.get(self.OBO_URL, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Download complete.")
            return temp_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def parse_ontology(self, raw_path: str):
        """Parses the raw OBO JSON into the CDSS-optimized format."""
        logger.info("Parsing ontology data...")
        hpo_dict = {}
        metadata = {
            "version": "unknown",
            "download_timestamp": datetime.now().isoformat(),
            "source": self.OBO_URL
        }

        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # OBO JSON structure: graphs[0].nodes
            graph = data.get("graphs", [{}])[0]
            nodes = graph.get("nodes", [])
            
            # Extract version from meta
            metadata["version"] = graph.get("meta", {}).get("version", "unknown")

            for node in nodes:
                node_id_full = node.get("id", "")
                # Only process HP terms (e.g., http://purl.obolibrary.org/obo/HP_0000001)
                if "/HP_" not in node_id_full:
                    continue

                hpo_id = node_id_full.split("/")[-1].replace("_", ":")
                name = node.get("lbl", "")
                
                synonyms = []
                meta = node.get("meta", {})
                if "synonyms" in meta:
                    for syn in meta["synonyms"]:
                        synonyms.append(syn.get("val", ""))

                hpo_dict[hpo_id] = {
                    "name": name,
                    "synonyms": list(set(synonyms)) # Deduplicate
                }

            # Save the optimized dictionary
            with open(self.dict_path, 'w', encoding='utf-8') as f:
                json.dump(hpo_dict, f, indent=2)
            
            # Save metadata
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Parsing successful. Processed {len(hpo_dict)} HPO terms.")
            
            # Cleanup raw file
            os.remove(raw_path)
            
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            raise

    def update(self):
        """Runs the full update cycle."""
        if not self.should_update():
            return

        try:
            raw_file = self.download_ontology()
            self.parse_ontology(raw_file)
            logger.info(f"HPO Dictionary updated successfully at {self.dict_path}")
        except Exception as e:
            logger.error(f"Update cycle failed: {e}")

if __name__ == "__main__":
    updater = HPOUpdater()
    updater.update()
