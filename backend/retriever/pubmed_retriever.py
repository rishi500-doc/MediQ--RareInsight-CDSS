"""
Rare Disease CDSS - PubMed Live Retriever
Queries NCBI PubMed E-utilities to fetch real-time biomedical abstracts
for any symptom set, bypassing the static ChromaDB knowledge base.

This is the fallback layer that ensures 100% coverage for any rare disease,
even if it is not yet in the local vector knowledge base.
"""

import httpx
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from backend.utils.common import get_logger

logger = get_logger("CDSS.PubMedRetriever")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TIMEOUT     = httpx.Timeout(10.0, connect=4.0)

# Use your own NCBI API key if available for higher rate limits (optional)
NCBI_API_KEY: Optional[str] = None   # set via os.getenv("NCBI_API_KEY") if desired


async def _get(url: str, params: dict) -> Optional[str]:
    """Generic async GET returning response text, or None on error."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.text
    except Exception as e:
        logger.warning(f"PubMed API error [{url}]: {e}")
        return None


async def search_pubmed_ids(query: str, max_results: int = 5) -> List[str]:
    """
    Searches PubMed and returns a list of PMIDs for the given query.
    """
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    raw = await _get(ESEARCH_URL, params)
    if not raw:
        return []

    try:
        import json
        data = json.loads(raw)
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning(f"PubMed esearch parse error: {e}")
        return []


async def fetch_pubmed_abstracts(pmids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches PubMed article titles and abstracts for given PMIDs.
    Returns a list of dicts with 'title', 'abstract', 'pmid'.
    """
    if not pmids:
        return []

    params = {
        "db":       "pubmed",
        "id":       ",".join(pmids),
        "retmode":  "xml",
        "rettype":  "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    raw = await _get(EFETCH_URL, params)
    if not raw:
        return []

    articles = []
    try:
        root = ET.fromstring(raw)
        for article in root.findall(".//PubmedArticle"):
            pmid_el  = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abs_els  = article.findall(".//AbstractText")

            pmid     = pmid_el.text if pmid_el is not None else "unknown"
            title    = title_el.text if title_el is not None else ""
            abstract = " ".join(
                (el.text or "") for el in abs_els if el.text
            )

            if abstract:
                articles.append({
                    "pmid":     pmid,
                    "title":    title,
                    "abstract": abstract
                })
    except Exception as e:
        logger.warning(f"PubMed efetch XML parse error: {e}")

    return articles


async def pubmed_retrieve(
    symptoms: List[str],
    genes: List[str] = None,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Full PubMed retrieval pipeline:
    1. Build a clinical query from symptoms + genes
    2. Search PubMed for matching PMIDs
    3. Fetch and return structured abstracts

    Returns results in the same schema as ChromaDB hybrid_results so they
    can be merged seamlessly into the retrieval pipeline.
    """
    genes = genes or []

    # Build a clean PubMed-optimized clinical query
    # Limit to 4 symptoms and 2 genes to avoid overly narrow/noisy queries
    clean_symptoms = [s.strip().strip('"') for s in symptoms[:4] if s.strip()]
    clean_genes    = [g.strip() for g in (genes or [])[:2] if g.strip()]

    query_parts = []
    if clean_symptoms:
        sym_terms = " OR ".join(f'"{s}"[All Fields]' for s in clean_symptoms)
        query_parts.append(f"({sym_terms})")
    if clean_genes:
        gene_terms = " OR ".join(f'"{g}"[Gene]' for g in clean_genes)
        query_parts.append(f"({gene_terms})")
    query_parts.append("(rare disease[MeSH] OR genetic disorder[MeSH])")

    query = " AND ".join(query_parts)
    logger.info(f"PubMed query: {query[:150]}...")

    pmids = await search_pubmed_ids(query, max_results=max_results)
    if not pmids:
        logger.info("PubMed returned no results for this query.")
        return []

    articles = await fetch_pubmed_abstracts(pmids)

    # Convert to hybrid_results schema
    results = []
    for art in articles:
        results.append({
            "disease":        art["title"],   # Best available label from PubMed
            "description":    art["abstract"],
            "source":         f"PubMed:{art['pmid']}",
            "semantic_score": 0.50,           # Neutral — will be rescored by reranker
            "hpo_score":      0.0,
            "keyword_score":  0.0,
            "hallmark_score": 0.0,
            "final_score":    0.50,
        })

    logger.info(f"PubMed retrieved {len(results)} articles.")
    return results
