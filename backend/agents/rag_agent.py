import io
import os
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import gc
import time
import hashlib
import re
from collections import defaultdict
import psutil  # For memory monitoring


class ResearchRAG:
    """
    Memory-optimized RAG with intelligent PDF extraction and semantic chunking
    """
    
    def __init__(
        self,
        max_memory_mb: int = 2048,
        max_pdf_size_mb: int = 10,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        # Local embedding model — no API key required
        print(f"[RAG] Loading embedding model: {embedding_model}")
        self._encoder = SentenceTransformer(embedding_model)
        self.vector_store: List[Dict[str, Any]] = []

        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_pdf_bytes = max_pdf_size_mb * 1024 * 1024
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_cache: Dict[str, np.ndarray] = {}

        print(f"[RAG] Initialized with {max_memory_mb}MB memory limit")
        print(f"[RAG] Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    # ==================== PUBLIC API ====================

    def ingest_papers(
        self, 
        papers_metadata: List[Dict[str, Any]],
        force_pdf_download: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest papers with intelligent PDF extraction and memory management
        
        Returns statistics about ingestion
        """
        self.vector_store = []
        self._embedding_cache.clear()
        
        if not papers_metadata:
            return self._empty_stats()
        
        print(f"[RAG] Processing {len(papers_metadata)} papers...")
        
        stats = {
            'total_papers': len(papers_metadata),
            'successful_pdfs': 0,
            'failed_pdfs': 0,
            'abstract_only': 0,
            'total_chunks': 0,
            'memory_usage_mb': 0
        }
        
        # Sort by citation count (process high-value papers first)
        sorted_papers = sorted(
            papers_metadata, 
            key=lambda p: p.get('citation_count', 0), 
            reverse=True
        )
        
        for idx, paper in enumerate(sorted_papers, 1):
            # Check memory before processing
            if not self._check_memory_available():
                print(f"[RAG] ⚠️  Memory limit reached. Processed {idx-1}/{len(papers_metadata)} papers.")
                break
            
            try:
                print(f"[RAG] [{idx}/{len(papers_metadata)}] Processing: {paper.get('title', 'Unknown')[:50]}...")
                
                chunks, is_full_text = self._process_single_paper(paper, force_pdf_download)
                
                if chunks:
                    self.vector_store.extend(chunks)
                    stats['total_chunks'] += len(chunks)
                    
                    if is_full_text:
                        stats['successful_pdfs'] += 1
                    else:
                        stats['abstract_only'] += 1
                else:
                    stats['failed_pdfs'] += 1
                
                # Cleanup after each paper
                gc.collect()
                
            except Exception as e:
                print(f"[RAG]  Error processing paper: {str(e)[:100]}")
                stats['failed_pdfs'] += 1
        
        # Final cleanup
        self._cleanup_duplicates()
        gc.collect()
        
        stats['memory_usage_mb'] = self._get_memory_usage_mb()
        stats['final_chunks'] = len(self.vector_store)
        
        print(f"[RAG]   Ingestion complete:")
        print(f"      - Full PDFs: {stats['successful_pdfs']}")
        print(f"      - Abstracts: {stats['abstract_only']}")
        print(f"      - Failed: {stats['failed_pdfs']}")
        print(f"      - Total chunks: {stats['final_chunks']}")
        print(f"      - Memory: {stats['memory_usage_mb']:.1f}MB")
        
        return stats

    def get_context(
        self,
        query: str,
        paper_id: Optional[str] = None,
        paper_ids: Optional[List[str]] = None,
        top_k: int = 8,
        diversity_penalty: float = 0.3,  # Reduce redundancy
        min_similarity: float = 0.3  # Filter low-quality matches
    ) -> str:
        """
        Retrieve contextually relevant chunks with diversity
        """
        if not self.vector_store or not query.strip():
            return ""
        
        try:
            # Get or compute query embedding
            q_emb = self._get_or_compute_embedding(query, is_query=True)
        except Exception as e:
            print(f"[RAG] Query embedding error: {e}")
            return ""
        
        # Filter candidates
        candidates = self._filter_candidates(paper_id, paper_ids)
        if not candidates:
            return ""
        
        # Compute similarity scores
        candidate_embeddings = np.stack([c["embedding"] for c in candidates])
        scores = self._cosine_scores(q_emb, candidate_embeddings)
        
        # Apply citation boost for abstract-only papers
        for i, candidate in enumerate(candidates):
            if not candidate.get("is_full_text", True):
                citation_boost = min(candidate.get("citation_count", 0) / 100 * 0.15, 0.15)
                scores[i] += citation_boost
        
        # Filter by minimum similarity
        valid_indices = np.where(scores >= min_similarity)[0]
        if len(valid_indices) == 0:
            return ""
        
        # Apply diversity (MMR - Maximal Marginal Relevance)
        selected_indices = self._mmr_selection(
            scores, 
            candidate_embeddings, 
            top_k, 
            diversity_penalty,
            valid_indices
        )
        
        # Build context with metadata
        context_parts = []
        for idx in selected_indices:
            candidate = candidates[idx]
            score = scores[idx]
            
            # Add metadata for better context
            title = candidate.get('title', 'Unknown')
            is_full = "📄" if candidate.get('is_full_text') else "📝"
            
            context_parts.append(
                f"{is_full} [{title}] (Relevance: {score:.2f})\n{candidate['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)

    def get_context_for_paper_set(
        self, 
        query: str, 
        paper_ids: List[str], 
        top_k: int = 8
    ) -> str:
        """Convenience method for multi-paper context"""
        return self.get_context(query=query, paper_ids=paper_ids, top_k=top_k)

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Detailed statistics about indexed papers"""
        if not self.vector_store:
            return self._empty_stats()
        
        paper_ids = set(c["paper_id"] for c in self.vector_store)
        full_text_ids = set(
            c["paper_id"] for c in self.vector_store if c.get("is_full_text", False)
        )
        
        # Calculate average chunks per paper
        chunks_per_paper = defaultdict(int)
        for chunk in self.vector_store:
            chunks_per_paper[chunk["paper_id"]] += 1
        
        avg_chunks = sum(chunks_per_paper.values()) / len(chunks_per_paper) if chunks_per_paper else 0
        
        return {
            "total_papers_indexed": len(paper_ids),
            "full_text_papers": len(full_text_ids),
            "abstract_only_papers": len(paper_ids) - len(full_text_ids),
            "total_chunks": len(self.vector_store),
            "avg_chunks_per_paper": round(avg_chunks, 1),
            "memory_usage_mb": self._get_memory_usage_mb(),
            "coverage_rate": f"{(len(full_text_ids) / len(paper_ids) * 100):.1f}%" if paper_ids else "0%"
        }

    # ==================== INTERNAL PROCESSING ====================

    def _process_single_paper(
        self, 
        paper: Dict[str, Any],
        force_pdf: bool
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Process a single paper with intelligent fallback strategy
        """
        paper_id = paper.get("paper_id")
        title = paper.get("title", "Unknown")
        citation_count = paper.get("citation_count", 0)
        
        # Strategy 1: Try full PDF if available
        full_text = None
        is_full_text = False
        
        if force_pdf:
            pdf_url = paper.get("pdf_url")
            if pdf_url and pdf_url != "N/A":
                full_text = self._download_and_extract_pdf(pdf_url, title)
                is_full_text = bool(full_text)
        
        # Strategy 2: Fallback to abstract + TLDR
        if not full_text:
            abstract = paper.get("abstract", "")
            tldr = paper.get("tldr", "")
            
            # Build rich abstract-based text
            full_text = self._build_abstract_text(paper)
            is_full_text = False
            
            if not full_text.strip():
                return [], False
        
        # Semantic chunking
        chunks = self._semantic_chunk_text(full_text, title)
        
        # Clear from memory
        del full_text
        gc.collect()
        
        # Embed chunks
        return self._embed_chunks(
            chunks,
            paper_id=paper_id,
            title=title,
            citation_count=citation_count,
            is_full_text=is_full_text
        ), is_full_text

    def _download_and_extract_pdf(self, url: str, title: str) -> str:
        """
        Memory-efficient PDF download with intelligent text extraction
        """
        if not url or url == "N/A":
            return ""
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf,*/*",
            "Referer": "https://scholar.google.com/",
        }
        
        try:
            # Stream with timeout
            resp = requests.get(
                url, 
                timeout=15,  # Reduced timeout
                headers=headers, 
                allow_redirects=True, 
                stream=True
            )
            resp.raise_for_status()
            
            # Verify content type
            content_type = resp.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'octet-stream' not in content_type:
                resp.close()
                return ""
            
            # Stream content with size limit
            content = b""
            for chunk in resp.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.max_pdf_bytes:
                    print(f"[RAG] PDF too large for '{title[:40]}' - skipping")
                    resp.close()
                    return ""
            
            resp.close()
            
            # Extract text efficiently
            reader = PdfReader(io.BytesIO(content))
            
            # Extract only from first N pages if too large
            max_pages = min(len(reader.pages), 50)  # Limit to 50 pages
            
            text_parts = []
            for page_num in range(max_pages):
                try:
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        # Clean extracted text
                        page_text = self._clean_pdf_text(page_text)
                        text_parts.append(page_text)
                except Exception as e:
                    print(f"[RAG] Page {page_num} extraction error: {e}")
                    continue
            
            text = "\n\n".join(text_parts)
            
            # Free memory
            del content, reader
            gc.collect()
            
            if text.strip():
                print(f"[RAG]  PDF extracted ({len(text)} chars): '{title[:40]}'")
                return text
            
            return ""
            
        except requests.exceptions.Timeout:
            print(f"[RAG] ⏱  PDF download timeout: '{title[:40]}'")
            return ""
        except requests.exceptions.HTTPError as e:
            print(f"[RAG] HTTP error {e.response.status_code}: '{title[:40]}'")
            return ""
        except Exception as e:
            print(f"[RAG] PDF error: {str(e)[:100]}")
            return ""

    def _build_abstract_text(self, paper: Dict[str, Any]) -> str:
        """
        Build rich text from paper metadata when PDF unavailable
        """
        parts = []
        
        title = paper.get("title", "")
        if title:
            parts.append(f"Title: {title}\n")
        
        authors = paper.get("authors", [])
        if authors:
            author_str = ", ".join(authors[:5])  # First 5 authors
            parts.append(f"Authors: {author_str}\n")
        
        year = paper.get("publication_year", "")
        if year:
            parts.append(f"Year: {year}\n")
        
        abstract = paper.get("abstract", "")
        if abstract:
            parts.append(f"\nAbstract:\n{abstract}\n")
        
        tldr = paper.get("tldr", "")
        if tldr:
            parts.append(f"\nTL;DR:\n{tldr}\n")
        
        keywords = paper.get("keywords", [])
        if keywords:
            parts.append(f"\nKeywords: {', '.join(keywords[:10])}\n")
        
        return "\n".join(parts)

    def _semantic_chunk_text(self, text: str, title: str) -> List[str]:
        """
        Intelligent semantic chunking that preserves context
        """
        # Clean text first
        text = self._clean_pdf_text(text)
        
        # Split by sections if available (common in academic papers)
        sections = self._split_by_sections(text)
        
        chunks = []
        for section in sections:
            # Further chunk if section is too large
            if len(section) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_by_sentences(section)
                chunks.extend(sub_chunks)
            else:
                if section.strip():
                    chunks.append(section)
        
        # Add title context to first chunk
        if chunks:
            chunks[0] = f"[{title}]\n\n{chunks[0]}"
        
        return chunks

    def _split_by_sections(self, text: str) -> List[str]:
        """
        Split text by common academic paper sections
        """
        # Common section headers
        section_patterns = [
            r'\n\s*(?:ABSTRACT|Abstract)\s*\n',
            r'\n\s*(?:INTRODUCTION|Introduction)\s*\n',
            r'\n\s*(?:RELATED WORK|Related Work|BACKGROUND|Background)\s*\n',
            r'\n\s*(?:METHODOLOGY|Methodology|METHODS|Methods)\s*\n',
            r'\n\s*(?:EXPERIMENTS|Experiments|RESULTS|Results)\s*\n',
            r'\n\s*(?:DISCUSSION|Discussion)\s*\n',
            r'\n\s*(?:CONCLUSION|Conclusion)\s*\n',
            r'\n\s*(?:REFERENCES|References)\s*\n',
        ]
        
        # Try to split by sections
        for pattern in section_patterns:
            if re.search(pattern, text):
                sections = re.split(pattern, text)
                # Filter out empty sections
                sections = [s.strip() for s in sections if s.strip()]
                if len(sections) > 1:
                    return sections
        
        # Fallback: split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        sections = []
        current_section = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if current_length + para_length > self.chunk_size * 2:
                if current_section:
                    sections.append('\n\n'.join(current_section))
                current_section = [para]
                current_length = para_length
            else:
                current_section.append(para)
                current_length += para_length
        
        if current_section:
            sections.append('\n\n'.join(current_section))
        
        return sections if sections else [text]

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences with overlap
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1]] if current_chunk else []
                    current_length = len(current_chunk[0]) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _clean_pdf_text(self, text: str) -> str:
        """
        Clean extracted PDF text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove references section (often not useful for RAG)
        ref_match = re.search(r'\n\s*(?:REFERENCES|References)\s*\n', text, re.IGNORECASE)
        if ref_match:
            text = text[:ref_match.start()]
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        
        return text.strip()

    def _embed_chunks(
        self,
        chunks: List[str],
        paper_id: str,
        title: str,
        citation_count: int,
        is_full_text: bool,
        batch_size: int = 5  # Smaller batches for stability
    ) -> List[Dict[str, Any]]:
        """
        Embed chunks with retry logic and caching
        """
        docs = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            if not batch:
                continue
            
            for attempt in range(3):
                try:
                    # Check if already cached
                    batch_embeddings = []
                    uncached_texts = []
                    uncached_indices = []
                    
                    for idx, text in enumerate(batch):
                        cache_key = self._get_cache_key(text)
                        if cache_key in self._embedding_cache:
                            batch_embeddings.append((idx, self._embedding_cache[cache_key]))
                        else:
                            uncached_texts.append(text)
                            uncached_indices.append(idx)
                    
                    # Embed uncached texts
                    if uncached_texts:
                        new_embeddings = self._encoder.encode(
                            uncached_texts, normalize_embeddings=True
                        )
                        
                        # Cache new embeddings
                        for text, emb, idx in zip(uncached_texts, new_embeddings, uncached_indices):
                            cache_key = self._get_cache_key(text)
                            emb_array = np.array(emb, dtype="float32")
                            self._embedding_cache[cache_key] = emb_array
                            batch_embeddings.append((idx, emb_array))
                    
                    # Sort by original index
                    batch_embeddings.sort(key=lambda x: x[0])
                    embeddings = [emb for _, emb in batch_embeddings]
                    
                    # Create documents
                    for text, emb in zip(batch, embeddings):
                        docs.append({
                            "text": text,
                            "embedding": emb,
                            "paper_id": paper_id,
                            "title": title,
                            "citation_count": citation_count,
                            "is_full_text": is_full_text,
                        })
                    
                    break  # Success
                    
                except Exception as e:
                    if attempt == 2:
                        print(f"[RAG] ❌ Embedding failed after 3 attempts: {str(e)[:100]}")
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff
        
        return docs

    def _get_or_compute_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Get embedding from cache or compute new one
        """
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Compute new embedding
        emb = np.array(
            self._encoder.encode(text, normalize_embeddings=True), dtype="float32"
        )
        self._embedding_cache[cache_key] = emb
        
        return emb

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text[:500].encode()).hexdigest()

    def _mmr_selection(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
        k: int,
        lambda_param: float,
        valid_indices: np.ndarray
    ) -> List[int]:
        """
        Maximal Marginal Relevance for diverse selection
        """
        selected = []
        candidates = set(valid_indices.tolist())
        
        # Select first (highest score)
        first_idx = valid_indices[np.argmax(scores[valid_indices])]
        selected.append(first_idx)
        candidates.remove(first_idx)
        
        # Select remaining with diversity
        for _ in range(min(k - 1, len(candidates))):
            if not candidates:
                break
            
            best_score = -float('inf')
            best_idx = None
            
            for idx in candidates:
                # Original relevance
                relevance = scores[idx]
                
                # Diversity (minimum similarity to already selected)
                similarities = [
                    self._cosine_similarity(embeddings[idx], embeddings[sel_idx])
                    for sel_idx in selected
                ]
                diversity = 1 - max(similarities) if similarities else 1
                
                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)
        
        return selected

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _filter_candidates(
        self,
        paper_id: Optional[str],
        paper_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Filter vector store by paper IDs"""
        if paper_id:
            return [c for c in self.vector_store if c["paper_id"] == paper_id]
        if paper_ids:
            paper_ids_set = set(paper_ids)
            return [c for c in self.vector_store if c["paper_id"] in paper_ids_set]
        return self.vector_store

    def _cosine_scores(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity scores"""
        q_norm = np.linalg.norm(query_emb) + 1e-8
        d_norms = np.linalg.norm(doc_embs, axis=1) + 1e-8
        return (doc_embs @ query_emb) / (d_norms * q_norm)

    def _cleanup_duplicates(self):
        """Remove duplicate chunks"""
        seen = set()
        unique_chunks = []
        
        for chunk in self.vector_store:
            chunk_hash = hashlib.md5(chunk['text'][:200].encode()).hexdigest()
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                unique_chunks.append(chunk)
        
        removed = len(self.vector_store) - len(unique_chunks)
        if removed > 0:
            print(f"[RAG] Removed {removed} duplicate chunks")
        
        self.vector_store = unique_chunks

    def _check_memory_available(self) -> bool:
        """Check if system has enough free RAM to continue"""
        try:
            free_mb = psutil.virtual_memory().available / 1024 / 1024
            return free_mb > 300  # stop only if less than 300MB free on the whole system
        except Exception:
            return True

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty stats"""
        return {
            "total_papers_indexed": 0,
            "full_text_papers": 0,
            "abstract_only_papers": 0,
            "total_chunks": 0,
            "avg_chunks_per_paper": 0,
            "memory_usage_mb": 0,
            "coverage_rate": "0%"
        }