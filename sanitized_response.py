# sanitized_response.py
"""
Enhanced response formatting with confidence-aware display
"""

from retriever import FAQRetriever
from config import SUPPORT_LINK, ICER_DOCS_BASE, DEBUG_MODE


# ============================================================================
# RESPONSE FORMATTERS
# ============================================================================

def format_high_confidence_response(faq_match):
    """
    Format response for high-confidence matches (clean, direct)
    
    Args:
        faq_match: FAQ result dictionary
    
    Returns:
        Formatted response string
    """
    response = f"""
✅ **Direct Answer** (High confidence)

**Q:** {faq_match['matched_question']}

**A:** {faq_match['matched_answer']}

---
📂 **Category:** {faq_match['category']}
🔗 **Source:** {faq_match['url']}
    """.strip()
    
    return response


def format_medium_confidence_response(faq_match):
    """
    Format response for medium-confidence matches
    
    Args:
        faq_match: FAQ result dictionary
    
    Returns:
        Formatted response string
    """
    response = f"""
🔍 **Best Match** (Medium confidence)

**Q:** {faq_match['matched_question']}

**A:** {faq_match['matched_answer']}

---
📂 **Category:** {faq_match['category']}
🔗 **Source:** {faq_match['url']}

💡 **Note:** This answer is likely relevant. If you need more specific information, please visit the source link above.
    """.strip()
    
    return response


def format_low_confidence_response(faq_match):
    """
    Format response for low-confidence matches (with disclaimer)
    
    Args:
        faq_match: FAQ result dictionary
    
    Returns:
        Formatted response string
    """
    response = f"""
⚠️  **Uncertain Match** (Low confidence)

**Potentially Related Question:**
{faq_match['matched_question']}

**Answer from Documentation:**
{faq_match['matched_answer']}

---
📂 **Category:** {faq_match['category']}
🔗 **Source:** {faq_match['url']}

---
⚠️  **Important:** This match may not fully answer your question. Please:
   • Review the source documentation: {faq_match['url']}
   • Submit a support ticket for specific help: {SUPPORT_LINK}
   • Browse ICER docs: {ICER_DOCS_BASE}
    """.strip()
    
    return response


def format_very_low_confidence_response(faq_match):
    """
    Format response for very low-confidence matches (strong disclaimer)
    
    Args:
        faq_match: FAQ result dictionary
    
    Returns:
        Formatted response string
    """
    response = f"""
❌ **No Confident Match Found**

I couldn't find a reliable answer in the ICER documentation for your question.

**The closest match I found was:**
{faq_match['matched_question']}

{faq_match['matched_answer'][:300]}{'...' if len(faq_match['matched_answer']) > 300 else ''}

---
🔗 **Source:** {faq_match['url']}

---
**🆘 Recommended Actions:**

1. **Submit a Support Ticket:** {SUPPORT_LINK}
   (ICER staff can provide accurate, personalized help)

2. **Browse Official Documentation:** {ICER_DOCS_BASE}
   (Search for keywords related to your question)

3. **Refine Your Question:** Try rephrasing with different keywords
    """.strip()
    
    return response


def format_response(faq_match):
    """
    Main formatter - routes to appropriate formatter based on confidence
    
    Args:
        faq_match: FAQ result dictionary with confidence level
    
    Returns:
        Formatted response string
    """
    confidence = faq_match.get('confidence', 'medium')
    
    if confidence == 'high':
        return format_high_confidence_response(faq_match)
    elif confidence == 'medium':
        return format_medium_confidence_response(faq_match)
    elif confidence == 'low':
        return format_low_confidence_response(faq_match)
    else:  # very_low
        return format_very_low_confidence_response(faq_match)


def format_multiple_results(faq_matches, show_top_n=3):
    """
    Format multiple FAQ results (useful for comparison)
    
    Args:
        faq_matches: List of FAQ result dictionaries
        show_top_n: Number of results to display
    
    Returns:
        Formatted response string
    """
    if not faq_matches:
        return f"""
❌ **No matches found**

Please try:
• Rephrasing your question
• Using different keywords
• Submitting a support ticket: {SUPPORT_LINK}
• Browsing ICER docs: {ICER_DOCS_BASE}
        """.strip()
    
    response_parts = [f"🔍 **Top {min(show_top_n, len(faq_matches))} Matches:**\n"]
    
    for i, match in enumerate(faq_matches[:show_top_n], 1):
        confidence_emoji = {
            'high': '✅',
            'medium': '🔍',
            'low': '⚠️',
            'very_low': '❌'
        }.get(match.get('confidence', 'medium'), '🔍')
        
        response_parts.append(f"""
{'-'*60}
**#{i}** {confidence_emoji} **{match['confidence'].upper()}**

**Q:** {match['matched_question']}

**A:** {match['matched_answer'][:200]}{'...' if len(match['matched_answer']) > 200 else ''}

📂 Category: {match['category']}
🔗 Source: {match['url']}
        """.strip())
    
    response_parts.append(f"\n{'-'*60}")
    response_parts.append(f"\n💡 **Tip:** For detailed answers, visit the source links above.")
    
    return "\n".join(response_parts)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def run_cli():
    """
    Interactive CLI for testing sanitized responses (no LLaMA)
    Fast direct responses only
    """
    print("\n" + "="*60)
    print("⚡ FAQ Direct Response Mode (No LLaMA)")
    print("="*60)
    print("\nThis mode shows direct FAQ matches without LLaMA synthesis.")
    print("It's FAST - perfect for high-confidence queries!\n")
    
    retriever = FAQRetriever(debug=DEBUG_MODE)
    
    print("💡 Commands:")
    print("  - Type a question to get direct FAQ match")
    print("  - 'top3' + question - Show top 3 matches")
    print("  - 'stats' - Show retriever statistics")
    print("  - 'quit' - Exit\n")
    
    while True:
        user_input = input("\n❓ Ask a question: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Exiting...")
            break
        
        if user_input.lower() == "stats":
            print(f"\n📊 **Statistics:**")
            print(f"  • Total FAQs: {len(retriever.faqs)}")
            print(f"  • Embedding cache: {retriever.embedding_cache}")
            print(f"  • Bi-encoder: {retriever.bi_encoder}")
            print(f"  • Cross-encoder: {retriever.cross_encoder}")
            continue
        
        if not user_input:
            continue
        
        # Check for 'top3' command
        show_multiple = False
        if user_input.lower().startswith('top3'):
            show_multiple = True
            user_input = user_input[4:].strip()
            if not user_input:
                print("❌ Please provide a question after 'top3'")
                continue
        
        # Get results
        print("\n" + "="*60)
        
        if show_multiple:
            results = retriever.find_top_k_faqs(user_input, k=3)
            formatted = format_multiple_results(results, show_top_n=3)
        else:
            results = retriever.find_top_k_faqs(user_input, k=1)
            if results:
                formatted = format_response(results[0])
            else:
                formatted = f"""
❌ **No matches found**

Please try:
• Rephrasing your question
• Using different keywords  
• Submitting a support ticket: {SUPPORT_LINK}
                """.strip()
        
        print(formatted)
        print("="*60)


def get_direct_answer(user_query, retriever=None):
    """
    Get formatted direct answer without LLaMA (for use in main.py)
    
    Args:
        user_query: User's question
        retriever: FAQRetriever instance (will create if None)
    
    Returns:
        Formatted response string
    """
    if retriever is None:
        retriever = FAQRetriever(debug=DEBUG_MODE)
    
    results = retriever.find_top_k_faqs(user_query, k=1)
    
    if not results:
        return f"""
❌ **No matches found in ICER documentation**

**Recommended Actions:**
1. Submit a support ticket: {SUPPORT_LINK}
2. Browse ICER docs: {ICER_DOCS_BASE}
3. Try rephrasing your question with different keywords
        """.strip()
    
    return format_response(results[0])


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    run_cli()
