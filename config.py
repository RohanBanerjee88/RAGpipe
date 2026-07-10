# config.py
"""
Centralized configuration for ICER FAQ system
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# SCRAPING CONFIGURATION
# ============================================================================

SCRAPE_URLS = [
    "https://docs.icer.msu.edu/Frequently_Asked_Questions_FAQ_/",
    "https://docs.icer.msu.edu/Linux_Shell/",
    "https://docs.icer.msu.edu/Submitting_a_Help_Ticket/",
    "https://docs.icer.msu.edu/Sensitive_Data_on_the_HPCC/"
]

# Add more URLs here as needed
# SCRAPE_URLS.append("https://docs.icer.msu.edu/new_page/")

SCRAPE_TIMEOUT = 10  # seconds
SCRAPE_MAX_WORKERS = 5  # concurrent threads
MIN_QUESTION_LENGTH = 10  # minimum characters for valid question (was 15)
MIN_ANSWER_LENGTH = 15  # minimum characters for valid answer (was 20)

# ============================================================================
# FILE PATHS
# ============================================================================

FAQ_JSON_PATH = PROJECT_ROOT / "all_faqs.json"
EMBEDDING_CACHE_PATH = PROJECT_ROOT / "faq_embeddings.pt"
SCRAPE_METADATA_PATH = PROJECT_ROOT / "scrape_metadata.json"  # for incremental updates

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Bi-encoder for initial semantic search
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"

# Cross-encoder for re-ranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLaMA model for complex queries
LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Number of candidates for bi-encoder stage
BI_ENCODER_TOP_K = 10

# Number of final results to return
FINAL_TOP_K = 3

# Hybrid retrieval: BM25 + semantic ranking fused before cross-encoder scoring
BM25_TOP_K = 20
SEMANTIC_TOP_K = 20
HYBRID_CANDIDATE_K = 20
RRF_RANK_CONSTANT = 60
RRF_BM25_WEIGHT = 1.0
RRF_SEMANTIC_WEIGHT = 1.0

# Final reranking combines normalized intent, keyword, and coverage signals.
# Absolute cross/semantic scores remain separate inputs to confidence routing.
FINAL_CROSS_WEIGHT = 1.0
FINAL_BM25_WEIGHT = 0.35
FINAL_LEXICAL_WEIGHT = 0.25

# ============================================================================
# ENSEMBLE CONFIDENCE SCORING (Multi-signal approach)
# ============================================================================

# Signal 1: Bi-encoder semantic similarity (cosine similarity, 0-1 range)
BI_ENCODER_THRESHOLDS = {
    "good_match": 0.60,      # Semantically similar
    "poor_match": 0.40       # Semantically distant
}

# Signal 2: Cross-encoder raw scores (model-specific, typically -10 to +10)
CROSS_ENCODER_RAW_THRESHOLDS = {
    "excellent": 5.0,        # Very confident match
    "good": 3.0,             # Decent match
    "poor": 1.0              # Weak match
}

# Signal 3: Cross-encoder normalized scores (0-1 range, relative ranking)
CROSS_ENCODER_NORMALIZED_THRESHOLDS = {
    "high": 0.75,            # Top candidate is clearly best
    "medium": 0.50,          # Moderate confidence
    "low": 0.30              # Low confidence
}

# Signal 4: Score gap between top 2 candidates (indicates clear winner)
SCORE_GAP_THRESHOLDS = {
    "clear_winner": 0.15,    # Significant gap = one clear answer
    "uncertain": 0.05        # Small gap = multiple similar matches
}

# Ensemble decision rules (how many signals must pass for each confidence level)
ENSEMBLE_RULES = {
    "high": 4,        # All 4 signals must pass
    "medium": 3,      # At least 3 signals pass
    "low": 2,         # At least 2 signals pass
    "very_low": 0     # Fewer than 2 signals (force LLaMA with strong disclaimer)
}

# Legacy thresholds (kept for backward compatibility, but ensemble overrides these)
CONFIDENCE_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.50,
    "low": 0.30
}
CROSS_ENCODER_RAW_THRESHOLD = 0.45

# Minimum evidence required before invoking an LLM. A query can pass through
# semantic/cross-encoder evidence or strong lexical evidence, but not relative
# ranking alone.
EVIDENCE_GATE = {
    "min_bi_score": 0.40,
    "min_cross_raw_score": 1.0,
    "min_bm25_normalized": 0.45,
    "min_lexical_coverage": 0.34,
}

# ============================================================================
# LLAMA CONFIGURATION
# ============================================================================

LLAMA_MAX_NEW_TOKENS = 200
LLAMA_TEMPERATURE = 0.7  # Lower = more deterministic
LLAMA_TOP_P = 0.9
LLAMA_DO_SAMPLE = True  # Set to False for fully deterministic output

# ============================================================================
# RESPONSE CONFIGURATION
# ============================================================================

# Support ticket link for when answers aren't found
SUPPORT_LINK = "https://contact.icer.msu.edu/contact"
ICER_DOCS_BASE = "https://docs.icer.msu.edu/"

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

DEBUG_MODE = False  # Set to True for verbose logging
ENABLE_CALIBRATION = False  # Set to True to run threshold calibration

# Retrieval tracing is opt-in because user questions may contain sensitive data.
TRACE_ENABLED = False
TRACE_PATH = PROJECT_ROOT / "logs" / "retrieval_traces.jsonl"

# Small in-process cache for repeated queries. This is intentionally not
# conversational memory and never changes factual retrieval behavior.
RETRIEVAL_CACHE_SIZE = 128

# ============================================================================
# TREE SEARCH CONFIGURATION (PageIndex-style)
# ============================================================================

TREE_JSON_PATH = PROJECT_ROOT / "faq_tree.json"
TREE_SEARCH_ENABLED = False         # Optional experiment; hybrid retrieval covers the current 54-FAQ corpus
TREE_SEARCH_MAX_NODES = 3           # Max categories to search
TREE_SEARCH_MAX_FAQS = 15           # Max FAQs to return from tree search
TREE_BUILD_MIN_SUBCATEGORY_SIZE = 5 # Min FAQs required to create subcategory
TREE_AUTO_UPDATE = True             # Auto-update tree after scraping

# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

# GPU/CPU configuration
USE_GPU = True  # Set to False to force CPU usage
