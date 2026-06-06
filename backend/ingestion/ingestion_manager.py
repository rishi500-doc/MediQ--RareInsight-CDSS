"""
Rare Disease CDSS - Ingestion Manager
Orchestrates async retrieval, chunking, and indexing across free data sources.
"""
import asyncio
from typing import List, Dict, Any
from backend.ingestion.fetcher import PubMedIngestor, OrphanetIngestor, ClinVarIngestor, HPOIngestor, MonarchIngestor, MedlinePlusIngestor
from backend.ingestion.chunker import BiomedicalChunker
from backend.ingestion.metadata_extractor import MetadataExtractor
from backend.ingestion.vector_indexer import VectorIndexer
from backend.ingestion.incremental_updater import IncrementalUpdater
from backend.utils.common import get_logger

logger = get_logger("CDSS.IngestionManager")

class IngestionManager:
    def __init__(self):
        self.pubmed = PubMedIngestor()
        self.orphanet = OrphanetIngestor()
        self.clinvar = ClinVarIngestor()
        self.hpo = HPOIngestor()
        self.monarch = MonarchIngestor()
        self.medlineplus = MedlinePlusIngestor()
        
        self.chunker = BiomedicalChunker()
        self.extractor = MetadataExtractor()
        self.indexer = VectorIndexer()
        self.updater = IncrementalUpdater()

    def _prepare_document(self, raw_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = self.chunker.chunk_text(raw_doc["raw_text"])
        ready_docs = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{raw_doc['document_id']}_chunk_{i}"
            if self.updater.is_indexed(doc_id):
                continue
                
            meta = raw_doc.get("metadata", {})
            meta = self.extractor.enrich(meta, chunk)
            
            # Format explicitly for required output schema
            formatted_meta = {
                "disease": meta.get("disease", ""),
                "genes": meta.get("genes", []),
                "hpo_terms": meta.get("hpo_terms", []),
                "pmid": meta.get("pmid", ""),
                "clinical_significance": meta.get("clinical_significance", ""),
                "inheritance_pattern": meta.get("inheritance_pattern", ""),
                "publication_date": meta.get("publication_date", "")
            }
            
            ready_docs.append({
                "document_id": doc_id,
                "source": raw_doc["source"],
                "title": raw_doc["title"],
                "chunk_text": chunk,
                "metadata": formatted_meta
            })
        return ready_docs

    async def ingest_pipeline(self):
        logger.info("Starting Multi-Source Biomedical Ingestion Pipeline...")
        
        # Concurrently fetch from free APIs
        pmids = await self.pubmed.search("rare genetic diseases", max_results=2)
        
        pubmed_task = self.pubmed.fetch(pmids)
        orphanet_task = self.orphanet.fetch_disease_dataset()
        clinvar_task = self.clinvar.search_variants("FBN1")
        hpo_task = self.hpo.fetch_annotations()
        monarch_task = self.monarch.fetch_disease_associations("MONDO:0007947")
        medlineplus_task = self.medlineplus.fetch_summaries()
        
        results = await asyncio.gather(
            pubmed_task, orphanet_task, clinvar_task, hpo_task, monarch_task, medlineplus_task
        )
        
        # Flatten all returned documents
        raw_docs = []
        for res in results:
            raw_docs.extend(res)
        
        all_chunks = []
        for doc in raw_docs:
            all_chunks.extend(self._prepare_document(doc))
            
        if all_chunks:
            await self.indexer.index_batch(all_chunks)
            for chunk in all_chunks:
                self.updater.mark_indexed(chunk["document_id"])
                
        logger.info(f"Ingestion Complete. {len(all_chunks)} chunks indexed into ChromaDB.")

if __name__ == "__main__":
    manager = IngestionManager()
    asyncio.run(manager.ingest_pipeline())
