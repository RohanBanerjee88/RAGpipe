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
from ingestion import content_hash, normalize_text, sha256_text, stable_source_id
from hybrid_retrieval import (
    BM25Index,
    expand_query,
    lexical_coverage,
    min_max_normalize,
    reciprocal_rank_fusion,
)
from observability import trace_event
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


class FAQRetriever:
    def __init__(self, debug=None):
        """
        Initialize FAQ retriever with bi-encoder and cross-encoder models
        
        Args:
            debug: Override DEBUG_MODE from config if specified
        """
        self.debug = debug if debug is not None else DEBUG_MODE
        
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
        
        with open(FAQ_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both old and new JSON formats
        if "faqs" in data:
            self.faqs = data["faqs"]
            self.metadata = data.get("metadata", {})
        else:
            # Old format compatibility
            self.faqs = data
            self.metadata = {}
        
        for faq in self.faqs:
            faq["question"] = normalize_text(faq.get("question"))
            faq["answer"] = normalize_text(faq.get("answer"))
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
        self.semantic_texts = [f"{faq['category']}. {faq['question']}" for faq in self.faqs]
        self.search_texts = [faq["search_text"] for faq in self.faqs]
        self.bm25 = BM25Index(self.search_texts)
        self.embedding_signature = sha256_text("\n".join(self.semantic_texts))

    def _load_models(self):
        """Load bi-encoder and cross-encoder models"""
        if self.debug:
            print(f"🤖 Loading bi-encoder: {BI_ENCODER_MODEL}...")
        
        device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        self.bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, device=device)
        
        if self.debug:
            print(f"🤖 Loading cross-encoder: {CROSS_ENCODER_MODEL}...")
        
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    def _load_embeddings(self):
        """Load or compute question embeddings"""
        if os.path.exists(EMBEDDING_CACHE_PATH):
            if self.debug:
                print(f"⚡ Loading cached embeddings from {EMBEDDING_CACHE_PATH}...")
            
            cached = torch.load(EMBEDDING_CACHE_PATH)
            if isinstance(cached, dict):
                self.question_embeddings = cached.get("embeddings")
                cached_signature = cached.get("signature")
            else:
                self.question_embeddings = cached
                cached_signature = None

            # Verify both corpus identity and embedding count. Count alone misses
            # changed content with the same number of FAQs.
            if (
                self.question_embeddings is None
                or len(self.question_embeddings) != len(self.semantic_texts)
                or cached_signature != self.embedding_signature
            ):
                if self.debug:
                    print("⚠️  Embedding cache is stale. Regenerating...")
                self._generate_embeddings()
        else:
            if self.debug:
                print("🔄 Generating new embeddings (this may take a moment)...")
            self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate and cache question embeddings"""
        self.question_embeddings = self.bi_encoder.encode(
            self.semantic_texts,
            convert_to_tensor=True,
            show_progress_bar=self.debug
        )
        torch.save({
            "signature": self.embedding_signature,
            "embeddings": self.question_embeddings,
        }, EMBEDDING_CACHE_PATH)
        
        if self.debug:
            print(f"💾 Embeddings cached to {EMBEDDING_CACHE_PATH}")

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
        # Evaluate each signal
        signals_passed = 0
        signal_details = {}
        
        # Signal 1: Bi-encoder semantic similarity
        bi_good = bi_score >= BI_ENCODER_THRESHOLDS["good_match"]
        signal_details["bi_encoder"] = {
            "value": bi_score,
            "passed": bi_good,
            "threshold": BI_ENCODER_THRESHOLDS["good_match"]
        }
        if bi_good:
            signals_passed += 1
        
        # Signal 2: Cross-encoder raw score
        cross_raw_good = raw_score >= CROSS_ENCODER_RAW_THRESHOLDS["good"]
        signal_details["cross_raw"] = {
            "value": raw_score,
            "passed": cross_raw_good,
            "threshold": CROSS_ENCODER_RAW_THRESHOLDS["good"]
        }
        if cross_raw_good:
            signals_passed += 1
        
        # Signal 3: Cross-encoder normalized score
        cross_norm_good = normalized_score >= CROSS_ENCODER_NORMALIZED_THRESHOLDS["high"]
        signal_details["cross_normalized"] = {
            "value": normalized_score,
            "passed": cross_norm_good,
            "threshold": CROSS_ENCODER_NORMALIZED_THRESHOLDS["high"]
        }
        if cross_norm_good:
            signals_passed += 1
        
        # Signal 4: Score gap (clear winner?)
        gap_good = score_gap >= SCORE_GAP_THRESHOLDS["clear_winner"]
        signal_details["score_gap"] = {
            "value": score_gap,
            "passed": gap_good,
            "threshold": SCORE_GAP_THRESHOLDS["clear_winner"]
        }
        if gap_good:
            signals_passed += 1

        # Relative signals alone are not enough evidence for a match. When both
        # absolute quality signals fail, the "best" candidate may simply be the
        # least bad item in the shortlist.
        absolute_match_plausible = bi_good or raw_score >= CROSS_ENCODER_RAW_THRESHOLDS["poor"]
        if not absolute_match_plausible:
            return "very_low", True, signal_details
        
        # Determine confidence level based on signals passed
        if signals_passed >= ENSEMBLE_RULES["high"]:
            confidence = "high"
            needs_llama = False
        elif signals_passed >= ENSEMBLE_RULES["medium"]:
            confidence = "medium"
            needs_llama = True
        elif signals_passed >= ENSEMBLE_RULES["low"]:
            confidence = "low"
            needs_llama = True
        else:
            confidence = "very_low"
            needs_llama = True
        
        return confidence, needs_llama, signal_details

    def _get_confidence_level(self, normalized_score):
        """
        Legacy confidence scoring (kept for backward compatibility)
        This is overridden by ensemble scoring when available
        
        Args:
            normalized_score: Score between 0-1
        
        Returns:
            tuple: (confidence_level, should_use_llama)
        """
        # Import legacy thresholds
        from config import CONFIDENCE_THRESHOLDS
        
        if normalized_score >= CONFIDENCE_THRESHOLDS["high"]:
            return "high", False
        elif normalized_score >= CONFIDENCE_THRESHOLDS["medium"]:
            return "medium", True
        elif normalized_score >= CONFIDENCE_THRESHOLDS["low"]:
            return "low", True
        else:
            return "very_low", True

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
        semantic_evidence = result["bi_score"] >= EVIDENCE_GATE["min_bi_score"]
        cross_evidence = result["raw_score"] >= EVIDENCE_GATE["min_cross_raw_score"]
        lexical_evidence = (
            result["bm25_normalized"] >= EVIDENCE_GATE["min_bm25_normalized"]
            and result["lexical_coverage"] >= EVIDENCE_GATE["min_lexical_coverage"]
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
        semantic_ranking = sorted(
            range(len(self.faqs)),
            key=lambda index: float(bi_scores[index]),
            reverse=True,
        )[:SEMANTIC_TOP_K]

        # Lexical ranking
        bm25_scores = self.bm25.scores(retrieval_query)
        bm25_normalized = min_max_normalize(bm25_scores)
        bm25_ranking = sorted(
            range(len(self.faqs)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )[:BM25_TOP_K]

        # Rank fusion avoids pretending cosine and BM25 scores share a scale.
        fused_scores = reciprocal_rank_fusion(
            [semantic_ranking, bm25_ranking],
            weights=[RRF_SEMANTIC_WEIGHT, RRF_BM25_WEIGHT],
            rank_constant=RRF_RANK_CONSTANT,
        )
        fused_ranking = sorted(fused_scores, key=fused_scores.get, reverse=True)[:HYBRID_CANDIDATE_K]

        if self.debug:
            print(f"  📊 Bi-encoder max score: {max(float(score) for score in bi_scores):.4f}")
            print(f"  🔤 BM25 max score: {max(bm25_scores, default=0.0):.4f}")
            print(f"  🔀 Hybrid candidates: {len(fused_ranking)}")

        # ====================================================================
        # STEP 2: Cross-encoder re-ranking of fused candidates
        # ====================================================================
        cross_inputs = [
            (user_query, f"{self.faqs[index]['category']}. {self.faqs[index]['question']}")
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
        normalized_scores = self._normalize_scores(cross_scores)
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
                "raw_score": candidate["raw_score"],
                "rerank_score": candidate["rerank_score"],
                "rerank_boost": candidate["rerank_boost"],
                "bm25_score": candidate["bm25_score"],
                "bm25_normalized": candidate["bm25_normalized"],
                "rrf_score": candidate["rrf_score"],
                "lexical_coverage": candidate["lexical_coverage"],
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
            if self._is_manipulative_query(user_query):
                result["evidence_sufficient"] = False
                result["evidence_rejection_reason"] = "source_manipulation_request"
            result["ensemble_signals"] = signal_details
            
            results.append(result)
        
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
        results = self.find_top_k_faqs(user_query, k=max(3, FINAL_TOP_K))
        
        if not results:
            return {
                "result": None,
                "route": "abstain",
                "reason": "no_matches"
            }
        
        best_match = results[0]
        
        if not best_match["evidence_sufficient"]:
            return {
                "result": best_match,
                "context_faqs": results[:3],
                "route": "abstain",
                "reason": "insufficient_evidence",
            }

        if best_match["needs_llama"]:
            return {
                "result": best_match,
                "context_faqs": results[:3],
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
        if os.path.exists(EMBEDDING_CACHE_PATH):
            os.remove(EMBEDDING_CACHE_PATH)
            print(f"🗑️  Deleted embedding cache: {EMBEDDING_CACHE_PATH}")


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
