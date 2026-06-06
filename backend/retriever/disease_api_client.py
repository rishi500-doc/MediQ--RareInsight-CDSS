"""
Rare Disease CDSS - Live Disease API Client
Fetches real-time phenotype and gene-disease association data from:

  1. HPO Ontology API (ontology.jax.org)  — disease → phenotypes + genes
     * NOTE: The old hpo.jax.org/api/hpo/ endpoints are DEAD (404).
     *        The API migrated to ontology.jax.org in 2025.
  2. NIH GARD API                         — rare disease metadata

All endpoints are free, public, and require no API key.
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from backend.utils.common import get_logger

logger = get_logger("CDSS.DiseaseAPIClient")

# ─── HPO API (new domain since 2025) ──────────────────────────────────────────
# Old (DEAD):  hpo.jax.org/api/hpo/search
# New (LIVE):  ontology.jax.org/api/network/search/disease
HPO_DISEASE_SEARCH  = "https://ontology.jax.org/api/network/search/disease"
HPO_ANNOTATION      = "https://ontology.jax.org/api/network/annotation"

GARD_SEARCH_URL     = "https://rarediseases.info.nih.gov/api/diseases/search"

TIMEOUT = httpx.Timeout(8.0, connect=4.0)


class DiseaseAPIClient:
    """
    Async client for live HPO disease-phenotype-gene data.
    Caches all results in-process to avoid redundant network calls.
    """

    def __init__(self):
        self._id_cache:        Dict[str, Optional[str]] = {}   # name → OMIM/ORPHA id
        self._phenotype_cache: Dict[str, List[str]]     = {}   # name → [phenotype strings]
        self._gene_cache:      Dict[str, List[str]]     = {}   # name → [gene symbols]

    # ─── Internal helpers ─────────────────────────────────────────────────────

    async def _get(self, url: str, params: dict = None) -> Optional[Any]:
        """Performs an async GET and returns parsed JSON, or None on error."""
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.get(url, params=params or {})
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.warning(f"API request failed [{url}]: {e}")
            return None

    # ─── HPO Disease ID resolution ────────────────────────────────────────────

    async def _search_disease_id(self, disease_name: str) -> Optional[str]:
        """
        Resolves a disease name to an HPO disease ID (OMIM:xxx or ORPHA:xxx).
        Uses the HPO ontology.jax.org search endpoint.
        """
        if disease_name in self._id_cache:
            return self._id_cache[disease_name]

        data = await self._get(HPO_DISEASE_SEARCH, {
            "q":   disease_name,
            "max": 5
        })

        disease_id = None
        if data and data.get("results"):
            # Prefer exact name match, then best-ranked
            for result in data["results"]:
                if result.get("name", "").lower() == disease_name.lower():
                    disease_id = result.get("id")
                    break
            if not disease_id:
                disease_id = data["results"][0].get("id")

        logger.debug(f"HPO disease ID for '{disease_name}': {disease_id}")
        self._id_cache[disease_name] = disease_id
        return disease_id

    # ─── Shared HPO annotation fetch ──────────────────────────────────────────

    async def _fetch_disease_data(self, disease_name: str) -> None:
        """
        Fetches the full HPO annotation for a disease.
        One call returns: disease info, phenotypes by category, AND genes.
        Populates both _phenotype_cache and _gene_cache.
        """
        if disease_name in self._phenotype_cache and disease_name in self._gene_cache:
            return  # already cached

        disease_id = await self._search_disease_id(disease_name)
        if not disease_id:
            self._phenotype_cache.setdefault(disease_name, [])
            self._gene_cache.setdefault(disease_name, [])
            return

        # Single HTTP call: GET /api/network/annotation/{diseaseId}
        # Returns: { disease: {...}, categories: { "Nervous System": [...], ... }, genes: [...] }
        data = await self._get(f"{HPO_ANNOTATION}/{disease_id}")
        if not data:
            self._phenotype_cache.setdefault(disease_name, [])
            self._gene_cache.setdefault(disease_name, [])
            return

        # ── Parse phenotypes from categories ──────────────────────────────
        phenotypes: List[str] = []
        categories = data.get("categories") or {}
        for category_name, terms in categories.items():
            for term in terms:
                name = (term.get("name") or "").strip().lower()
                if name and name not in phenotypes:
                    phenotypes.append(name)
        self._phenotype_cache[disease_name] = phenotypes

        # ── Parse genes ───────────────────────────────────────────────────
        genes: List[str] = []
        for g in data.get("genes") or []:
            symbol = (g.get("name") or "").strip()
            if symbol and symbol not in genes:
                genes.append(symbol)
        self._gene_cache[disease_name] = genes

        logger.info(
            f"HPO API → '{disease_name}' [{disease_id}]: "
            f"{len(phenotypes)} phenotypes, {len(genes)} genes"
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    async def get_disease_phenotypes(self, disease_name: str) -> List[str]:
        """Returns a list of lowercase clinical phenotype names for a disease."""
        if disease_name not in self._phenotype_cache:
            await self._fetch_disease_data(disease_name)
        return self._phenotype_cache.get(disease_name, [])

    async def get_disease_genes(self, disease_name: str) -> List[str]:
        """
        Returns gene symbols associated with a disease from HPO.

        Example:
            get_disease_genes("Huntington disease") → ["HTT"]
            get_disease_genes("Marfan syndrome")    → ["FBN1"]
        """
        if disease_name not in self._gene_cache:
            await self._fetch_disease_data(disease_name)
        genes = self._gene_cache.get(disease_name, [])
        logger.info(f"Genes for '{disease_name}': {genes[:8]}")
        return genes

    # ─── HPO term → disease reverse lookup ────────────────────────────────────

    async def get_diseases_by_hpo_term(self, hpo_id: str) -> List[Dict[str, Any]]:
        """
        Reverse HPO lookup: given an HPO term ID (e.g. HP:0002072),
        return diseases associated with it via HPO annotation API.
        """
        data = await self._get(f"{HPO_ANNOTATION}/{hpo_id}")
        if not data or not data.get("diseases"):
            return []

        diseases = []
        for d in data["diseases"]:
            did   = d.get("id", "")
            dname = d.get("name", "Unknown")
            if did and dname:
                diseases.append({"diseaseId": did, "diseaseName": dname})
        return diseases

    async def get_diseases_by_hpo_terms(
        self,
        hpo_ids: List[str],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Reverse HPO lookup for a patient: given a list of HPO term IDs,
        fetch all associated diseases in parallel, then rank by overlap count.
        """
        if not hpo_ids:
            return []

        results = await asyncio.gather(
            *[self.get_diseases_by_hpo_term(hpo_id) for hpo_id in hpo_ids]
        )

        # Count overlap per disease
        overlap: Dict[str, Dict[str, Any]] = {}
        for diseases in results:
            for d in diseases:
                did  = d.get("diseaseId", "")
                name = d.get("diseaseName", "Unknown")
                if did not in overlap:
                    overlap[did] = {"disease_id": did, "name": name, "count": 0}
                overlap[did]["count"] += 1

        ranked = sorted(overlap.values(), key=lambda x: x["count"], reverse=True)[:top_n]

        candidates = []
        for item in ranked:
            hpo_score = round(item["count"] / max(len(hpo_ids), 1), 3)
            candidates.append({
                "disease":        item["name"],
                "disease_id":     item["disease_id"],
                "description":    f"HPO: {item['count']} of {len(hpo_ids)} patient phenotypes match.",
                "source":         "HPO-Direct",
                "semantic_score": 0.0,
                "hpo_score":      hpo_score,
                "keyword_score":  0.0,
                "hallmark_score": 0.0,
                "final_score":    hpo_score,
                "hpo_overlap":    item["count"],
            })

        logger.info(
            f"HPO reverse lookup: {len(hpo_ids)} terms → "
            f"{len(overlap)} unique diseases → top {len(candidates)} returned"
        )
        return candidates

    async def get_gard_info(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """Fetches disease metadata from NIH GARD."""
        data = await self._get(GARD_SEARCH_URL, {
            "query":    disease_name,
            "type":     "diseases",
            "pageSize": 1
        })
        if data and data.get("items"):
            return data["items"][0]
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────
disease_api_client = DiseaseAPIClient()
