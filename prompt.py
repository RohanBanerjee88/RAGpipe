# prompt.py
"""
Smart LLaMA handler with persistent loading and confidence-aware prompts
"""

import os

from transformers import AutoConfig, pipeline
import torch
from retriever import FAQRetriever
from grounding import (
    INSUFFICIENT_EVIDENCE,
    build_evidence_blocks,
    extractive_grounded_answer,
    format_sources,
    validate_grounded_answer,
)
from sanitized_response import format_response
from config import (
    LLAMA_MODEL,
    LLAMA_MAX_NEW_TOKENS,
    LLAMA_TEMPERATURE,
    LLAMA_TOP_P,
    LLAMA_DO_SAMPLE,
    SUPPORT_LINK,
    ICER_DOCS_BASE,
    DEBUG_MODE,
    USE_GPU
)


# ============================================================================
# GLOBAL LLAMA INSTANCE (Persistent across queries)
# ============================================================================

_llama_pipeline = None
_llama_load_attempted = False
_llama_task = None


def _get_model_name():
    """Return the configured LLM model, allowing local test overrides."""
    return os.getenv("FAQ_LLM_MODEL", LLAMA_MODEL)


def _get_pipeline_task(model_name):
    """Pick the correct generation pipeline for causal vs seq2seq models."""
    task_override = os.getenv("FAQ_LLM_TASK")
    if task_override:
        return task_override

    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        return "text2text-generation"

    return "text-generation"


def get_llama_pipeline():
    """
    Get or initialize LLaMA pipeline (lazy loading with persistence)
    Only loads once per session, then reuses the same instance
    
    Returns:
        HuggingFace text-generation pipeline
    """
    global _llama_pipeline, _llama_load_attempted, _llama_task
    
    if _llama_pipeline is not None:
        return _llama_pipeline
    
    if _llama_load_attempted:
        # Already tried and failed
        raise RuntimeError("LLaMA model failed to load previously")
    
    _llama_load_attempted = True
    
    try:
        print("\n" + "="*60)
        model_name = _get_model_name()
        _llama_task = _get_pipeline_task(model_name)

        print(f"🔄 Loading LLM model: {model_name}")
        print(f"🧰 Pipeline task: {_llama_task}")
        print("⏳ Please wait...")
        print("="*60 + "\n")
        
        pipeline_kwargs = {
            "task": _llama_task,
            "model": model_name,
        }

        if USE_GPU and torch.cuda.is_available():
            pipeline_kwargs["device_map"] = "auto"
            pipeline_kwargs["torch_dtype"] = torch.float16
        else:
            pipeline_kwargs["device"] = -1

        _llama_pipeline = pipeline(**pipeline_kwargs)
        
        print("\n" + "="*60)
        print("✅ LLM model loaded successfully!")
        print("🚀 Subsequent queries will be faster")
        print("="*60 + "\n")
        
        return _llama_pipeline
        
    except Exception as e:
        print(f"\n❌ Failed to load LLaMA model: {e}")
        print("💡 Falling back to direct FAQ responses only\n")
        _llama_pipeline = None
        raise


def unload_llama():
    """
    Unload LLaMA from memory (useful for freeing GPU/RAM)
    """
    global _llama_pipeline, _llama_load_attempted, _llama_task
    
    if _llama_pipeline is not None:
        del _llama_pipeline
        _llama_pipeline = None
        _llama_task = None
        _llama_load_attempted = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🗑️  LLaMA model unloaded from memory")


def is_llama_loaded():
    """Check if LLaMA is currently loaded"""
    return _llama_pipeline is not None


def get_llama_task():
    """Return the active generation task, if the model is loaded."""
    return _llama_task


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_high_confidence_prompt(user_query, top_faq):
    """
    Build prompt for high-confidence matches (usually not used with LLaMA)
    This is a fallback if high-confidence still goes to LLaMA
    
    Args:
        user_query: User's question
        top_faq: Single best-matching FAQ
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert assistant for ICER (Institute for Cyber-Enabled Research) at Michigan State University.

The user asked: "{user_query}"

Here is a highly relevant FAQ from ICER's official documentation:

Q: {top_faq['matched_question']}
A: {top_faq['matched_answer']}

Based on this information, provide a clear, direct answer to the user's question. If the FAQ fully answers the question, you can rephrase it naturally. Keep your response concise and helpful.

Answer:"""
    return prompt.strip()


def build_medium_confidence_prompt(user_query, top_faqs):
    """
    Build prompt for medium-confidence matches
    
    Args:
        user_query: User's question
        top_faqs: List of top matching FAQs (typically 3)
    
    Returns:
        Formatted prompt string
    """
    context_blocks = []
    for i, faq in enumerate(top_faqs, 1):
        context_blocks.append(
            f"[FAQ {i}] (Relevance: {faq['normalized_score']:.2f})\n"
            f"Q: {faq['matched_question']}\n"
            f"A: {faq['matched_answer']}\n"
            f"Source: {faq['url']}\n"
            f"Category: {faq['category']}"
        )
    
    context = "\n\n".join(context_blocks)
    
    prompt = f"""You are an expert assistant for ICER (Institute for Cyber-Enabled Research) at Michigan State University.

The user asked: "{user_query}"

Here are the most relevant FAQs from ICER's documentation:

{context}

Instructions:
1. Use the FAQ information above to answer the user's question
2. Synthesize information from multiple FAQs if needed
3. If the FAQs don't fully answer the question, provide the closest relevant information
4. Keep your response clear, concise, and helpful
5. You can mention which FAQ(s) you're drawing from

Answer:"""
    return prompt.strip()


def build_low_confidence_prompt(user_query, top_faqs):
    """
    Build prompt for low-confidence matches (with disclaimer)
    
    Args:
        user_query: User's question
        top_faqs: List of top matching FAQs (typically 3)
    
    Returns:
        Formatted prompt string
    """
    context_blocks = []
    for i, faq in enumerate(top_faqs, 1):
        context_blocks.append(
            f"[Related FAQ {i}] (Relevance: {faq['normalized_score']:.2f})\n"
            f"Q: {faq['matched_question']}\n"
            f"A: {faq['matched_answer']}\n"
            f"Category: {faq['category']}"
        )
    
    context = "\n\n".join(context_blocks)
    
    prompt = f"""You are an expert assistant for ICER (Institute for Cyber-Enabled Research) at Michigan State University.

The user asked: "{user_query}"

⚠️ IMPORTANT: The confidence for this query is LOW. The FAQs below may not directly answer the question.

Here are potentially related FAQs:

{context}

Instructions:
1. Be HONEST about uncertainty - start your response by acknowledging that you're not entirely certain
2. Provide the most relevant information you can from the FAQs
3. Suggest that the user consult ICER's official documentation or submit a support ticket
4. Keep your response helpful but cautious
5. Include this support link: {SUPPORT_LINK}

Answer (start with a disclaimer):"""
    return prompt.strip()


def build_grounded_prompt(user_query, context_faqs, max_sources=3):
    """Build a source-constrained prompt with stable citation identifiers."""
    evidence, sources = build_evidence_blocks(context_faqs, max_sources=max_sources)
    prompt = f"""You answer questions about ICER using only the evidence below.

User question: {user_query}

Evidence:
{evidence}

Rules:
1. Use only facts explicitly stated in the evidence.
2. Do not use outside knowledge, assumptions, or invented steps.
3. Put an inline citation like [S1] after every factual claim.
4. Cite only source IDs that appear above.
5. If the evidence does not answer the question, output exactly: {INSUFFICIENT_EVIDENCE}
6. Keep the answer concise and practical.

Answer:"""
    return prompt.strip(), sources


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_llama_response(prompt, confidence_level="medium"):
    """
    Generate response using LLaMA with appropriate parameters
    
    Args:
        prompt: Formatted prompt string
        confidence_level: 'high', 'medium', or 'low'
    
    Returns:
        Generated text response
    """
    try:
        llm = get_llama_pipeline()
        
        # Adjust generation parameters based on confidence
        temperature = LLAMA_TEMPERATURE
        if confidence_level == "high":
            temperature = 0.3  # More deterministic for high confidence
        elif confidence_level == "low":
            temperature = 0.7  # Allow more creativity for synthesis
        
        if DEBUG_MODE:
            print(f"\n🤖 Generating response (confidence: {confidence_level})...")
        
        generation_kwargs = {
            "max_new_tokens": LLAMA_MAX_NEW_TOKENS,
            "temperature": temperature,
            "top_p": LLAMA_TOP_P,
            "do_sample": LLAMA_DO_SAMPLE,
        }

        if get_llama_task() == "text2text-generation":
            generation_kwargs["do_sample"] = False
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)
            generation_kwargs["max_length"] = generation_kwargs.pop("max_new_tokens")

        if getattr(llm, "tokenizer", None) is not None and llm.tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = llm.tokenizer.eos_token_id

        # Small local smoke-test models commonly have 512-token context
        # windows. Production models can accept the same flag harmlessly.
        output = llm(prompt, truncation=True, **generation_kwargs)
        
        generated_text = output[0]["generated_text"]
        
        # Extract answer part (remove prompt)
        if get_llama_task() == "text2text-generation":
            answer = generated_text.strip()
        elif "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            # Fallback: get everything after the prompt
            answer = generated_text[len(prompt):].strip()
        
        return answer
        
    except Exception as e:
        print(f"\n❌ Error generating LLaMA response: {e}")
        return None


def get_answer_with_llama(user_query, retriever=None):
    """Get an answer through the same evidence gate used by the main assistant."""
    if retriever is None:
        retriever = FAQRetriever(debug=DEBUG_MODE)
    
    if DEBUG_MODE:
        print(f"\n🔍 Query: {user_query}")
    
    # Get routing decision from retriever
    decision = retriever.get_best_match(user_query)
    
    if not decision["result"] or decision["route"] == "abstain":
        return (
            "I could not find enough evidence in the indexed ICER documentation "
            "to answer this reliably.\n\n"
            f"Documentation: {ICER_DOCS_BASE}\nSupport: {SUPPORT_LINK}"
        )

    result = decision["result"]
    confidence = result["confidence"]
    
    if DEBUG_MODE:
        print(f"📊 Confidence: {confidence}")
        print(f"🎯 Score: {result['normalized_score']:.3f}")
    
    if decision["route"] == "direct":
        return format_response(result)

    context_faqs = decision.get("context_faqs", [result])
    prompt, sources = build_grounded_prompt(user_query, context_faqs)
    response = generate_llama_response(prompt, confidence_level=confidence)

    if response is None:
        response = extractive_grounded_answer(sources[0], "generation_error")
        return f"{response}\n\n{format_sources(sources[:1])}"

    valid, reason = validate_grounded_answer(response, len(sources))
    if response.strip() == INSUFFICIENT_EVIDENCE:
        return (
            "I could not find enough evidence in the indexed ICER documentation "
            "to answer this reliably."
        )
    if not valid:
        response = extractive_grounded_answer(sources[0], reason)
        return f"{response}\n\n{format_sources(sources[:1])}"

    return f"{response}\n\n{format_sources(sources)}"


# ============================================================================
# CLI Testing Interface
# ============================================================================

def test_prompt_system():
    """Interactive CLI for testing the complete system"""
    print("\n" + "="*60)
    print("🧪 LLaMA + Retriever Test Mode")
    print("="*60)
    
    retriever = FAQRetriever(debug=DEBUG_MODE)
    
    print("\n💡 Commands:")
    print("  - Type a question to get an answer")
    print("  - 'load' - Pre-load LLaMA model")
    print("  - 'unload' - Unload LLaMA from memory")
    print("  - 'status' - Check LLaMA status")
    print("  - 'quit' - Exit\n")
    
    while True:
        user_input = input("\n❓ You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Exiting...")
            break
        
        if user_input.lower() == "load":
            try:
                get_llama_pipeline()
                print("✅ LLaMA is now loaded and ready")
            except Exception as e:
                print(f"❌ Failed to load LLaMA: {e}")
            continue
        
        if user_input.lower() == "unload":
            unload_llama()
            continue
        
        if user_input.lower() == "status":
            if is_llama_loaded():
                print("✅ LLaMA is currently loaded in memory")
            else:
                print("❌ LLaMA is not loaded")
            continue
        
        if not user_input:
            continue
        
        # Get answer
        print("\n" + "="*60)
        answer = get_answer_with_llama(user_input, retriever)
        print("\n🤖 AI:", answer)
        print("="*60)


if __name__ == "__main__":
    test_prompt_system()
