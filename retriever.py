# retriever.py
"""
Enhanced FAQ retriever with intelligent confidence scoring and smart routing
"""

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import json
import torch
import os
import re
from config import (
    FAQ_JSON_PATH,
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    EMBEDDING_CACHE_PATH,
    BI_ENCODER_TOP_K,
    FINAL_TOP_K,
    BI_ENCODER_THRESHOLDS,
    CROSS_ENCODER_RAW_THRESHOLDS,
    CROSS_ENCODER_NORMALIZED_THRESHOLDS,
    SCORE_GAP_THRESHOLDS,
    ENSEMBLE_RULES,
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
        
        self.questions = [faq["question"] for faq in self.faqs]

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
            
            self.question_embeddings = torch.load(EMBEDDING_CACHE_PATH)
            
            # Verify embedding count matches FAQ count
            if len(self.question_embeddings) != len(self.questions):
                if self.debug:
                    print("⚠️  Embedding count mismatch. Regenerating...")
                self._generate_embeddings()
        else:
            if self.debug:
                print("🔄 Generating new embeddings (this may take a moment)...")
            self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate and cache question embeddings"""
        self.question_embeddings = self.bi_encoder.encode(
            self.questions,
            convert_to_tensor=True,
            show_progress_bar=self.debug
        )
        torch.save(self.question_embeddings, EMBEDDING_CACHE_PATH)
        
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

        for token in shared:
            if len(token) >= 2 and token.upper() == token and any(char.isalpha() for char in token):
                boost += 2.0

        return boost

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
        
        if self.debug:
            print(f"\n🔍 Searching for: '{user_query}'")

        # ====================================================================
        # STEP 1: Bi-encoder semantic search (fast, broad filter)
        # ====================================================================
        device = self.question_embeddings.device
        query_embedding = self.bi_encoder.encode(
            user_query,
            convert_to_tensor=True
        ).to(device)

        # Compute cosine similarity
        bi_scores = util.pytorch_cos_sim(query_embedding, self.question_embeddings)[0]
        
        # Get top candidates for cross-encoder
        scored = [
            (score.item(), idx, self.faqs[idx]) 
            for idx, score in enumerate(bi_scores)
        ]
        top_bi_candidates = sorted(scored, key=lambda x: x[0], reverse=True)[:BI_ENCODER_TOP_K]

        if self.debug:
            max_bi_score = max([s[0] for s in top_bi_candidates])
            print(f"  📊 Bi-encoder max score: {max_bi_score:.4f}")

        # ====================================================================
        # STEP 2: Cross-encoder re-ranking (precise, but slower)
        # ====================================================================
        cross_inputs = [
            (user_query, faq["question"]) 
            for _, _, faq in top_bi_candidates
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

        for (bi_score, _, faq), raw_score, normalized_score in zip(
            top_bi_candidates,
            cross_scores,
            normalized_scores
        ):
            rerank_boost = self._lexical_rerank_boost(user_query, faq)
            reranked_candidates.append({
                "faq": faq,
                "bi_score": float(bi_score),
                "raw_score": float(raw_score),
                "normalized_score": float(normalized_score),
                "rerank_score": float(raw_score) + rerank_boost,
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
                "normalized_score": candidate["normalized_score"],
                "bi_score": candidate["bi_score"],
                "score_gap": float(score_gap),
                "confidence": confidence_level,
                "needs_llama": needs_llama,
                "matched_question": faq["question"],
                "matched_answer": faq["answer"],
                "url": faq.get("url", "N/A"),
                "category": faq.get("category", "General"),
                "hash": faq.get("hash", "N/A")
            }
            
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

        # Return all for calibration, or top-k for normal use
        if return_all_candidates:
            return results
        
        # Filter only on calibrated ensemble confidence. Normalized scores are
        # relative within the candidate set and can make irrelevant queries look
        # artificially strong.
        filtered_results = [
            r for r in results 
            if r["confidence"] != "very_low"
        ]

        # If no results pass threshold, return top-k with very_low confidence
        if not filtered_results:
            if self.debug:
                print("  ⚠️  No results above threshold. Returning top-k with low confidence.")
            
            return results[:k]

        return filtered_results[:k]

    def get_best_match(self, user_query):
        """
        Get single best match with routing decision
        
        Args:
            user_query: User's question
        
        Returns:
            dict with 'result' and 'route' ('direct' or 'llama')
        """
        results = self.find_top_k_faqs(user_query, k=1)
        
        if not results:
            return {
                "result": None,
                "route": "llama",
                "reason": "no_matches"
            }
        
        best_match = results[0]
        
        if best_match["needs_llama"]:
            # Get top 3 for LLaMA context
            top_3 = self.find_top_k_faqs(user_query, k=3)
            return {
                "result": best_match,
                "context_faqs": top_3,
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
            print(f"  Confidence thresholds: {CONFIDENCE_THRESHOLDS}")
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
