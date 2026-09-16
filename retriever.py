# retriever.py
"""
Enhanced FAQ retriever with intelligent confidence scoring and smart routing
"""

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import json
import torch
import os
import re
import copy
import time
from collections import OrderedDict
from functools import lru_cache
from model_setup import retrieval_location
from knowledge_store import list_collections
from embedding_store import cached_embeddings, model_fingerprint
from ingestion import content_hash, normalize_text, sha256_text, stable_source_id
from hybrid_retrieval import (
    BM25Index,
    expand_query,
    lexical_coverage,
    lexical_overlap_count,
    min_max_normalize,
    reciprocal_rank_fusion,
)
from observability import trace_event
from device_runtime import select_runtime_device
from source_confidence import (
    combined_source_confidence,
    consistency_score,
    freshness_score,
    trust_score,
)
from config import (
    FAQ_JSON_PATH,
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    EMBEDDING_CACHE_PATH,
    FINAL_TOP_K,
    BM25_TOP_K,
    SEMANTIC_TOP_K,
    HYBRID_CANDIDATE_K,
    RRF_RANK_CONSTANT,
    RRF_BM25_WEIGHT,
    RRF_SEMANTIC_WEIGHT,
    FINAL_CROSS_WEIGHT,
    FINAL_BM25_WEIGHT,
    FINAL_LEXICAL_WEIGHT,
    BI_ENCODER_THRESHOLDS,
    CROSS_ENCODER_RAW_THRESHOLDS,
    CROSS_ENCODER_NORMALIZED_THRESHOLDS,
    SCORE_GAP_THRESHOLDS,
    ENSEMBLE_RULES,
    EVIDENCE_GATE,
    RETRIEVAL_CACHE_SIZE,
    DEBUG_MODE,
    USE_GPU
)


@lru_cache(maxsize=4)
def shared_models(bi_location, cross_location, device, bi_fingerprint, cross_fingerprint):
    return (
        SentenceTransformer(bi_location, device=device, local_files_only=True),
        CrossEncoder(cross_location, device=device, local_files_only=True),
    )


class FAQRetriever:
    def __init__(self, debug=None, collection=None):
        """
        Initialize FAQ retriever with bi-encoder and cross-encoder models
        
        Args:
            debug: Override DEBUG_MODE from config if specified
        """
        self.debug = debug if debug is not None else DEBUG_MODE
        self.collection = collection or os.getenv("FAQ_COLLECTION", "auto")
        
        # Load FAQ data
        self._load_faqs()
        
        # Load models
        self._load_models()
        
        # Load or compute embeddings
        self._load_embeddings()
        self._result_cache = OrderedDict()
        
        if self.debug:
            print(f"✅ Retriever initialized with {len(self.faqs)} FAQs")

    def _load_faqs(self):
        """Load FAQ data from JSON"""
        if self.debug:
            print(f"📂 Loading FAQs from {FAQ_JSON_PATH}...")
        
        collections = list_collections()
        if self.collection != "auto":
            collections = [item for item in collections if item["name"] == self.collection]
            if not collections:
                raise ValueError(f"Collection {self.collection!r} is missing or empty")
        self.faqs = [record for item in collections for record in item["records"]]
        if not self.faqs:
            raise ValueError("No searchable collections. Run scrape.py or collections import first.")
        self.metadata = {}
        self.collection_indices = {}
        for index, faq in enumerate(self.faqs):
            self.collection_indices.setdefault(faq["collection"], []).append(index)
        
        for faq in self.faqs:
            faq["question"] = normalize_text(faq.get("question"))
            faq["answer"] = faq.get("text", faq.get("answer", ""))
            faq["category"] = normalize_text(faq.get("category") or "General")
            faq["section"] = normalize_text(faq.get("section") or faq["category"])
            faq["search_text"] = normalize_text(
                faq.get("search_text") or f"{faq['category']}. {faq['question']}. {faq['answer']}"
            )
            faq["source_id"] = faq.get("source_id") or stable_source_id(
                faq.get("url", ""), faq["category"], faq["question"]
            )
            faq["content_hash"] = faq.get("content_hash") or content_hash(
                faq["question"], faq["answer"]
            )
            faq["version"] = int(faq.get("version", 1))

        self.questions = [faq["question"] for faq in self.faqs]
        self.semantic_texts = [
            f"{faq['category']}. {faq['question']}" if faq.get("record_type") == "faq"
            else f"{faq['title'][:120]}. {faq['text']}" for faq in self.faqs
        ]
        self.search_texts = [faq["search_text"] for faq in self.faqs]
        self.collection_bm25 = {
            name: BM25Index(self.search_texts[index] for index in indices)
            for name, indices in self.collection_indices.items()
        }
        self.embedding_signature = sha256_text("\n".join(self.semantic_texts))

    def _load_models(self):
        """Load bi-encoder and cross-encoder models"""
        if self.debug:
            print(f"🤖 Loading bi-encoder: {BI_ENCODER_MODEL}...")
        
        self.device_selection = select_runtime_device(USE_GPU)
        self.device = self.device_selection.device
        if self.debug:
            print(f"🖥️  Model device: {self.device_selection.summary()}")
        bi_location = retrieval_location(BI_ENCODER_MODEL)
        cross_location = retrieval_location(CROSS_ENCODER_MODEL)
        self.model_fingerprint = model_fingerprint(bi_location)
        self.bi_encoder, self.cross_encoder = shared_models(
            bi_location, cross_location, self.device, self.model_fingerprint,
            model_fingerprint(cross_location),
        )
        
        if self.debug:
            print(f"🤖 Loading cross-encoder: {CROSS_ENCODER_MODEL}...")
        

    def _load_embeddings(self):
        """Load or compute question embeddings"""
        self.embedded_passages = 0
        parts = []
        for name, indices in self.collection_indices.items():
            embeddings, count = cached_embeddings(
                name, [self.semantic_texts[index] for index in indices], self.bi_encoder,
                self.model_fingerprint, self.device,
            )
            parts.append(embeddings)
            self.embedded_passages += count
        self.question_embeddings = torch.cat(parts)

    def _normalize_scores(self, scores):
        """
        Normalize scores to 0-1 range for consistent thresholding
        
        Args:
            scores: List or numpy array of raw scores
        
        Returns:
            List of normalized scores (0-1)
        """
        # Handle empty scores (works with both lists and numpy arrays)
        if len(scores) == 0:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score - min_score < 1e-5:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _get_confidence_level_ensemble(self, bi_score, raw_score, normalized_score, score_gap):
        """
        Ensemble confidence scoring using multiple signals
        
        Args:
            bi_score: Bi-encoder cosine similarity (0-1)
            raw_score: Cross-encoder raw score (model-specific range)
            normalized_score: Cross-encoder normalized score (0-1)
            score_gap: Gap between top and second score (0-1)
        
        Returns:
            tuple: (confidence_level, needs_llama, signal_breakdown)
        """
        # Confidence uses absolute pair scores; candidate-set normalization is ranking only.
        details = {
            "bi_encoder": {"value": bi_score, "passed": bi_score >= 0.60, "threshold": 0.60},
            "cross_raw": {"value": raw_score, "passed": raw_score >= 3.0, "threshold": 3.0},
        }
        if bi_score >= 0.60 and raw_score >= 3.0:
            return "high", False, details
        if bi_score >= 0.60 or raw_score >= 1.0:
            return "medium", True, details
        return "very_low", True, details

    def _lexical_rerank_boost(self, user_query, faq):
        """
        Add a small reranking boost for exact technical token matches.
        Confidence still uses raw model scores; this only breaks cases where the
        cross-encoder prefers generic phrasing over a specific acronym/error.
        """
        query_tokens = set(re.findall(r"[A-Za-z0-9_]+", user_query))
        faq_text = f"{faq.get('question', '')} {faq.get('category', '')}"
        faq_tokens = set(re.findall(r"[A-Za-z0-9_]+", faq_text))

        shared = query_tokens & faq_tokens
        boost = 0.15 * len({token.lower() for token in shared if len(token) >= 3})

        discriminative_acronyms = {"OOM", "GPU", "CPU", "SSH", "SFTP", "SLURM", "CUDA"}
        for token in shared:
            if token in discriminative_acronyms:
                boost += 2.0

        return boost

    def _has_sufficient_evidence(self, result):
        """Require an absolute semantic/cross signal or strong lexical evidence."""
        if result.get("record_type", "faq") != "faq":
            return (result["raw_score"] >= 1.0 and result["bi_score"] >= 0.30
                    and result["lexical_overlap_count"] >= 1)
        semantic_evidence = result["bi_score"] >= EVIDENCE_GATE["min_bi_score"]
        cross_evidence = result["raw_score"] >= EVIDENCE_GATE["min_cross_raw_score"]
        lexical_evidence = (
            result["bm25_normalized"] >= EVIDENCE_GATE["min_bm25_normalized"]
            and result["lexical_coverage"] >= EVIDENCE_GATE["min_lexical_coverage"]
            and result["lexical_overlap_count"] >= EVIDENCE_GATE["min_lexical_overlap_terms"]
        )
        return semantic_evidence or cross_evidence or lexical_evidence

    def _is_manipulative_query(self, user_query):
        """Reject requests to invent, override, or misrepresent source evidence."""
        patterns = (
            r"\bpretend\b.*\b(faq|documentation|source)\b",
            r"\bmake up\b.*\b(faq|documentation|source)\b",
            r"\bfabricate\b",
            r"\bignore\b.*\b(documentation|evidence|source|instructions)\b",
            r"\breveal\b.*\bsystem prompt\b",
        )
        lowered = user_query.lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    def find_top_k_faqs(self, user_query, k=None, return_all_candidates=False):
        """
        Find top-k most relevant FAQs using bi-encoder + cross-encoder
        
        Args:
            user_query: User's question
            k: Number of results to return (default: FINAL_TOP_K from config)
            return_all_candidates: If True, return all scored candidates for calibration
        
        Returns:
            List of FAQ matches with scores and metadata
        """
        if k is None:
            k = FINAL_TOP_K

        started_at = time.perf_counter()
        user_query = normalize_text(user_query)
        cache_key = user_query.lower()
        cached_results = self._result_cache.get(cache_key)
        if cached_results is not None:
            self._result_cache.move_to_end(cache_key)
            results = copy.deepcopy(cached_results)
            trace_event("retrieval_cache_hit", {
                "query": user_query,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            })
            return results if return_all_candidates else results[:k]
        
        if self.debug:
            print(f"\n🔍 Searching for: '{user_query}'")

        retrieval_query = expand_query(user_query)

        # ====================================================================
        # STEP 1: Independent semantic and BM25 retrieval
        # ====================================================================
        device = self.question_embeddings.device
        query_embedding = self.bi_encoder.encode(
            user_query,
            convert_to_tensor=True
        ).to(device)

        # Semantic ranking
        bi_scores = util.pytorch_cos_sim(query_embedding, self.question_embeddings)[0]
        semantic_ranking, bm25_ranking, fused_ranking = [], [], []
        bm25_scores, bm25_normalized = [0.0] * len(self.faqs), [0.0] * len(self.faqs)
        fused_scores = {}
        for name, indices in self.collection_indices.items():
            local_bi = bi_scores[indices]
            semantic = [indices[i] for i in torch.topk(local_bi, min(SEMANTIC_TOP_K, len(indices))).indices.tolist()]
            lexical = self.collection_bm25[name].scores(retrieval_query)
            normalized = min_max_normalize(lexical)
            keyword = [indices[i] for i in sorted(range(len(indices)), key=lexical.__getitem__, reverse=True)[:BM25_TOP_K]]
            for index, score, norm in zip(indices, lexical, normalized):
                bm25_scores[index], bm25_normalized[index] = score, norm
            fused = reciprocal_rank_fusion([semantic, keyword], weights=[RRF_SEMANTIC_WEIGHT, RRF_BM25_WEIGHT], rank_constant=RRF_RANK_CONSTANT)
            semantic_ranking.extend(semantic)
            bm25_ranking.extend(keyword)
            fused_scores.update(fused)
            fused_ranking.extend(sorted(fused, key=fused.get, reverse=True)[:HYBRID_CANDIDATE_K])
        bi_scores = bi_scores.cpu().tolist()

        if self.debug:
            print(f"  📊 Bi-encoder max score: {max(float(score) for score in bi_scores):.4f}")
            print(f"  🔤 BM25 max score: {max(bm25_scores, default=0.0):.4f}")
            print(f"  🔀 Hybrid candidates: {len(fused_ranking)}")

        # ====================================================================
        # STEP 2: Cross-encoder re-ranking of fused candidates
        # ====================================================================
        cross_inputs = [
            (user_query, self.semantic_texts[index])
            for index in fused_ranking
        ]
        cross_scores = self.cross_encoder.predict(cross_inputs)

        if self.debug:
            max_cross = max(cross_scores)
            min_cross = min(cross_scores)
            print(f"  📊 Cross-encoder range: [{min_cross:.4f}, {max_cross:.4f}]")

        # ====================================================================
        # STEP 3: Re-rank by cross-encoder, then score confidence
        # ====================================================================
        normalized_scores = [0.0] * len(cross_scores)
        for name in self.collection_indices:
            positions = [i for i, index in enumerate(fused_ranking) if self.faqs[index]["collection"] == name]
            norms = self._normalize_scores([cross_scores[i] for i in positions])
            for i, norm in zip(positions, norms):
                normalized_scores[i] = norm
        reranked_candidates = []

        for faq_index, raw_score, normalized_score in zip(
            fused_ranking,
            cross_scores,
            normalized_scores
        ):
            faq = self.faqs[faq_index]
            rerank_boost = self._lexical_rerank_boost(retrieval_query, faq)
            coverage = max(
                lexical_coverage(user_query, self.search_texts[faq_index]),
                lexical_coverage(retrieval_query, self.search_texts[faq_index]),
            )
            overlap_count = max(
                lexical_overlap_count(user_query, self.search_texts[faq_index]),
                lexical_overlap_count(retrieval_query, self.search_texts[faq_index]),
            )
            rerank_score = (
                FINAL_CROSS_WEIGHT * float(normalized_score)
                + FINAL_BM25_WEIGHT * float(bm25_normalized[faq_index])
                + FINAL_LEXICAL_WEIGHT * coverage
                + rerank_boost
            )
            reranked_candidates.append({
                "faq_index": faq_index,
                "faq": faq,
                "bi_score": float(bi_scores[faq_index]),
                "bm25_score": float(bm25_scores[faq_index]),
                "bm25_normalized": float(bm25_normalized[faq_index]),
                "rrf_score": float(fused_scores[faq_index]),
                "lexical_coverage": coverage,
                "lexical_overlap_count": overlap_count,
                "raw_score": float(raw_score),
                "normalized_score": float(normalized_score),
                "rerank_score": rerank_score,
                "rerank_boost": rerank_boost
            })

        reranked_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)

        # ====================================================================
        # STEP 4: Ensemble confidence scoring with multiple signals
        # ====================================================================
        results = []
        for idx, candidate in enumerate(reranked_candidates):
            next_normalized_score = (
                reranked_candidates[idx + 1]["normalized_score"]
                if idx < len(reranked_candidates) - 1
                else candidate["normalized_score"]
            )
            score_gap = max(0.0, candidate["normalized_score"] - next_normalized_score)

            confidence_level, needs_llama, signal_details = self._get_confidence_level_ensemble(
                bi_score=candidate["bi_score"],
                raw_score=candidate["raw_score"],
                normalized_score=candidate["normalized_score"],
                score_gap=score_gap
            )

            faq = candidate["faq"]
            result = {
                "collection": faq["collection"],
                "record_type": faq.get("record_type", "faq"),
                "location": faq.get("location", ""),
                "imported_at": faq.get("imported_at"),
                "fetched_at": faq.get("fetched_at"),
                "raw_score": candidate["raw_score"],
                "rerank_score": candidate["rerank_score"],
                "rerank_boost": candidate["rerank_boost"],
                "bm25_score": candidate["bm25_score"],
                "bm25_normalized": candidate["bm25_normalized"],
                "rrf_score": candidate["rrf_score"],
                "lexical_coverage": candidate["lexical_coverage"],
                "lexical_overlap_count": candidate["lexical_overlap_count"],
                "normalized_score": candidate["normalized_score"],
                "bi_score": candidate["bi_score"],
                "score_gap": float(score_gap),
                "confidence": confidence_level,
                "needs_llama": needs_llama,
                "matched_question": faq["question"],
                "matched_answer": faq["answer"],
                "url": faq.get("url", "N/A"),
                "category": faq.get("category", "General"),
                "section": faq.get("section", faq.get("category", "General")),
                "source_id": faq.get("source_id", "N/A"),
                "scraped_at": faq.get("scraped_at", "N/A"),
                "version": faq.get("version", 1),
                "hash": faq.get("content_hash", faq.get("hash", "N/A"))
            }
            result["trust_score"] = trust_score(result["url"])
            result["freshness_score"] = freshness_score(result["scraped_at"])
            result["retrieval_consistency"] = consistency_score(
                candidate["faq_index"], semantic_ranking, bm25_ranking
            )
            result["source_confidence"] = combined_source_confidence(
                result["trust_score"],
                result["freshness_score"],
                result["retrieval_consistency"],
            )
            result["evidence_sufficient"] = self._has_sufficient_evidence(result)
            if result["record_type"] != "faq":
                result["needs_llama"] = True
                result["confidence"] = "medium" if result["evidence_sufficient"] else "very_low"
            if self._is_manipulative_query(user_query):
                result["evidence_sufficient"] = False
                result["evidence_rejection_reason"] = "source_manipulation_request"
            result["ensemble_signals"] = signal_details
            
            results.append(result)
        
        # Preserve within-collection FAQ ranking, then select collections by absolute evidence.
        if len(self.collection_indices) > 1:
            grouped = {name: [r for r in results if r["collection"] == name] for name in self.collection_indices}
            def collection_score(name):
                top = grouped[name][0]
                return (top["evidence_sufficient"], top["raw_score"] + 4 * top["bi_score"])
            order = sorted(grouped, key=collection_score, reverse=True)
            mentioned = [name for name in order if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", user_query, re.I)]
            if len(mentioned) == 1:
                order = mentioned + [name for name in order if name not in mentioned]
            results = [r for name in order for r in grouped[name]]

        # Debug output for ensemble signals (top result only)
        if self.debug and results and "ensemble_signals" in results[0]:
            print(f"\n  🎯 Ensemble Signals for Top Result:")
            signals = results[0]["ensemble_signals"]
            for signal_name, signal_data in signals.items():
                status = "✅" if signal_data["passed"] else "❌"
                print(f"    {status} {signal_name}: {signal_data['value']:.3f} (threshold: {signal_data['threshold']:.3f})")
            
            signals_passed = sum(1 for s in signals.values() if s["passed"])
            print(f"  📊 Signals Passed: {signals_passed}/4")
            print(f"  🏆 Final Confidence: {results[0]['confidence'].upper()}")

        trace_event("retrieval", {
            "query": user_query,
            "candidate_count": len(results),
            "candidates": [
                {
                    "rank": rank,
                    "source_id": result["source_id"],
                    "question": result["matched_question"],
                    "category": result["category"],
                    "bi_score": result["bi_score"],
                    "bm25_score": result["bm25_score"],
                    "bm25_normalized": result["bm25_normalized"],
                    "rrf_score": result["rrf_score"],
                    "cross_raw_score": result["raw_score"],
                    "rerank_score": result["rerank_score"],
                    "lexical_coverage": result["lexical_coverage"],
                    "lexical_overlap_count": result["lexical_overlap_count"],
                    "confidence": result["confidence"],
                    "evidence_sufficient": result["evidence_sufficient"],
                    "source_confidence": result["source_confidence"],
                }
                for rank, result in enumerate(results[:10], start=1)
            ],
        })

        trace_event("retrieval_latency", {
            "query": user_query,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "cache_hit": False,
        })

        self._result_cache[cache_key] = copy.deepcopy(results)
        self._result_cache.move_to_end(cache_key)
        while len(self._result_cache) > RETRIEVAL_CACHE_SIZE:
            self._result_cache.popitem(last=False)

        # Return all for calibration, or top-k for normal use. Do not discard a
        # top candidate merely because it is uncertain; routing handles that.
        if return_all_candidates:
            return results

        return results[:k]

    def get_best_match(self, user_query):
        """
        Get single best match with routing decision
        
        Args:
            user_query: User's question
        
        Returns:
            dict with 'result' and 'route' ('direct' or 'llama')
        """
        results = self.find_top_k_faqs(user_query, return_all_candidates=True)
        
        if not results:
            return {
                "result": None,
                "route": "abstain",
                "reason": "no_matches"
            }
        
        best_match = results[0]
        if (best_match.get("record_type") != "faq" and
                re.search(r"\b(calculate|compute|average|median|correlation|sum)\b", user_query, re.I)):
            return {"result": best_match, "route": "abstain", "reason": "dataset_analysis_not_supported"}
        supported = [r for r in results if r["evidence_sufficient"]]
        tops = {}
        for result in supported:
            tops.setdefault(result["collection"], result)
        multi = bool(re.search(r"\b(compare|both|across|versus)\b", user_query, re.I))
        overlapping_sources = [r for r in supported
            if r["collection"] == best_match["collection"]
            and r["matched_question"].lower() == best_match["matched_question"].lower()
            and r["url"] != best_match["url"]
            and r["matched_answer"] != best_match["matched_answer"]]
        if overlapping_sources and not multi:
            return {"result": best_match, "route": "clarify", "reason": "ambiguous_sources",
                    "collections": [best_match["collection"]],
                    "context_faqs": [best_match, overlapping_sources[0]]}
        if len(tops) > 1:
            first, second = list(tops.values())[:2]
            ambiguous = abs((first["raw_score"] + 4 * first["bi_score"]) -
                            (second["raw_score"] + 4 * second["bi_score"])) < 1.5
            ambiguous |= (first["matched_question"].lower() == second["matched_question"].lower()
                          and first["matched_answer"] != second["matched_answer"])
            if any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", user_query, re.I) for name in tops):
                ambiguous = False
            if ambiguous and not multi:
                return {"result": first, "route": "clarify", "reason": "ambiguous_collections",
                        "collections": list(tops), "context_faqs": list(tops.values())[:5]}
            if multi:
                return {"result": first, "route": "llama", "reason": "multiple_collections",
                        "context_faqs": list(tops.values())[:5], "multiple_collections": True}
        
        if not best_match["evidence_sufficient"]:
            return {
                "result": best_match,
                "context_faqs": [r for r in results if r["evidence_sufficient"]][:3],
                "route": "abstain",
                "reason": "insufficient_evidence",
            }

        if best_match["needs_llama"]:
            return {
                "result": best_match,
                "context_faqs": [r for r in results if r["evidence_sufficient"]][:3],
                "route": "llama",
                "reason": f"confidence_{best_match['confidence']}"
            }
        else:
            return {
                "result": best_match,
                "route": "direct",
                "reason": "high_confidence"
            }

    def invalidate_cache(self):
        """Delete embedding cache to force regeneration"""
        self._result_cache.clear()
        from local_store import data_root
        from filelock import FileLock
        for name in self.collection_indices:
            path = data_root() / "collections" / name / "embeddings.pt"
            with FileLock(str(path) + ".lock"):
                path.unlink(missing_ok=True)
        self._load_embeddings()


# ============================================================================
# CLI Testing Interface
# ============================================================================

def test_retriever():
    """Interactive CLI for testing retriever"""
    print("\n" + "="*60)
    print("🧪 FAQ Retriever Test Mode")
    print("="*60)
    
    retriever = FAQRetriever(debug=True)
    
    print("\n💡 Type 'quit' to exit, 'stats' for statistics\n")
    
    while True:
        query = input("\n❓ Ask a question: ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print("👋 Exiting...")
            break
        
        if query.lower() == "stats":
            print(f"\n📊 Statistics:")
            print(f"  Total FAQs: {len(retriever.faqs)}")
            print(f"  Embedding cache: {EMBEDDING_CACHE_PATH}")
            print(f"  Confidence thresholds: {BI_ENCODER_THRESHOLDS}")
            continue
        
        if not query:
            continue
        
        # Get routing decision
        decision = retriever.get_best_match(query)
        
        print(f"\n{'='*60}")
        print(f"🎯 Route: {decision['route'].upper()}")
        print(f"📍 Reason: {decision['reason']}")
        print(f"{'='*60}")
        
        if decision['result']:
            result = decision['result']
            print(f"\n🏆 Best Match:")
            print(f"  Confidence: {result['confidence'].upper()}")
            print(f"  Score: {result['normalized_score']:.3f}")
            print(f"  Question: {result['matched_question']}")
            print(f"  Answer: {result['matched_answer'][:200]}...")
            print(f"  Category: {result['category']}")
            print(f"  URL: {result['url']}")
            
            if decision['route'] == 'llama' and 'context_faqs' in decision:
                print(f"\n📚 Additional context for LLaMA ({len(decision['context_faqs'])} FAQs):")
                for i, faq in enumerate(decision['context_faqs'], 1):
                    print(f"  {i}. [{faq['normalized_score']:.3f}] {faq['matched_question'][:60]}...")
        else:
            print("❌ No matches found")


if __name__ == "__main__":
    test_retriever()
