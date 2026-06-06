import os
import re
import xml.etree.ElementTree as ET
import asyncio
import httpx
from typing import List, Dict, Any
from backend.utils.common import get_logger

logger = get_logger("CDSS.Ingestion")


class ClinVarIngestor:
    async def search_variants(self, gene: str) -> List[Dict[str, Any]]:
        # Mock API logic for ClinVar as esearch/efetch for clinvar is complex XML
        await asyncio.sleep(0.1)
        return [{
            "document_id": f"CLINVAR_{gene}_1",
            "source": "ClinVar",
            "title": f"Pathogenic variant in {gene}",
            "raw_text": f"Variant c.123A>G in {gene} is classified as Pathogenic.",
            "metadata": {
                "genes": [gene],
                "clinical_significance": "Pathogenic"
            }
        }]

class HPOIngestor:
    async def fetch_annotations(self) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.1)
        # Parses phenotype.hpoa file or similar text dumps
        return [{
            "document_id": "HPO_1",
            "source": "HPO",
            "title": "HPO Phenotype annotations",
            "raw_text": "HP:0000001 All associated phenotypes and terms.",
            "metadata": {
                "hpo_terms": ["HP:0000001"]
            }
        }]

class MonarchIngestor:
    async def fetch_disease_associations(self, disease_id: str) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.1)
        return [{
            "document_id": f"MONARCH_{disease_id}",
            "source": "Monarch Initiative",
            "title": f"Associations for {disease_id}",
            "raw_text": f"Phenotypic associations provided by Monarch for {disease_id}.",
            "metadata": {
                "disease": disease_id
            }
        }]

class OrphanetIngestor:
    async def fetch_disease_dataset(self) -> List[Dict[str, Any]]:
        docs = []
        try:
            # Production would parse the downloaded large XML streams from Orphadata.
            await asyncio.sleep(0.1)
            docs.append({
                "document_id": "ORPHA_100",
                "source": "Orphanet",
                "title": "Rare Disease XML Dump",
                "raw_text": "Orphanet mapping linking diseases to HPO traits.",
                "metadata": {
                    "disease": "Example Rare Disease",
                    "inheritance_pattern": "Autosomal Dominant"
                }
            })
        except Exception as e:
            logger.error(f"Orphanet ingest error: {e}")
        return docs

class PubMedIngestor:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async def search(self, query: str, max_results: int = 50) -> List[str]:
        url = f"{self.BASE_URL}/esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax={max_results}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])

    async def fetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids: return []
        ids_str = ",".join(pmids)
        url = f"{self.BASE_URL}/efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
        docs = []
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=30.0)
                root = ET.fromstring(resp.text)
                for article in root.findall(".//PubmedArticle"):
                    pmid = article.findtext(".//PMID")
                    title = article.findtext(".//ArticleTitle") or "Unknown Title"
                    abstract = "".join([elem.text for elem in article.findall(".//AbstractText") if elem.text])
                    if abstract:
                        docs.append({
                            "document_id": f"PMID_{pmid}",
                            "source": "PubMed",
                            "title": title,
                            "raw_text": abstract,
                            "metadata": {
                                "pmid": pmid,
                                "publication_date": article.findtext(".//PubDate/Year") or "Unknown"
                            }
                        })
        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")
        return docs

class MedlinePlusIngestor:
    async def fetch_summaries(self) -> List[Dict[str, Any]]:
        """Fetches consumer-friendly genetic summaries from MedlinePlus (NIH)."""
        logger.info("Fetching from MedlinePlus Genetics...")
        docs = []
        slugs = [
            "cystic-fibrosis", "marfan-syndrome", "huntington-disease",
            "sickle-cell-disease", "tay-sachs-disease", "achondroplasia",
            "down-syndrome", "fragile-x-syndrome", "noonan-syndrome"
        ]
        
        async with httpx.AsyncClient() as client:
            for slug in slugs:
                try:
                    url = f"https://medlineplus.gov/download/genetics/condition/{slug}.json"
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        data = response.json()
                        summary = data.get("health-condition-summary", {}).get("content", "")
                        genetics = data.get("genetic-changes", {}).get("content", "")
                        clean_desc = re.sub('<[^<]+?>', '', f"{summary} {genetics}")
                        
                        disease_title = data.get("title", slug.replace("-", " ").title())
                        docs.append({
                            "document_id": f"MEDLINEPLUS_{slug.upper()}",
                            "source": "MedlinePlus",
                            "title": disease_title,
                            "raw_text": clean_desc[:1000],
                            "metadata": {
                                "disease": disease_title
                            }
                        })
                    await asyncio.sleep(0.2)
                except Exception as e:
                    logger.error(f"MedlinePlus error for '{slug}': {e}")
        return docs
