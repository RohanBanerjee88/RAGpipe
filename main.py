# main.py
"""
Smart FAQ Assistant Router
Decides between fast direct responses and LLaMA synthesis based on confidence
"""

from retriever import FAQRetriever
from sanitized_response import format_response
from prompt import get_llama_pipeline, is_llama_loaded, generate_llama_response
from prompt import build_high_confidence_prompt, build_medium_confidence_prompt, build_low_confidence_prompt
from config import SUPPORT_LINK, ICER_DOCS_BASE, DEBUG_MODE
import sys


# ============================================================================
# SMART ROUTING LOGIC
# ============================================================================

class SmartFAQAssistant:
    """
    Intelligent FAQ assistant with adaptive routing
    """
    
    def __init__(self, debug=None):
        """
        Initialize the assistant
        
        Args:
            debug: Override DEBUG_MODE from config if specified
        """
        self.debug = debug if debug is not None else DEBUG_MODE
        self.retriever = FAQRetriever(debug=self.debug)
        self.llama = None
        self.stats = {
            "total_queries": 0,
            "direct_responses": 0,
            "llama_responses": 0,
            "failed_queries": 0
        }
        
        if self.debug:
            print("✅ Smart FAQ Assistant initialized")
    
    def _load_llama_if_needed(self):
        """Load LLaMA model if not already loaded"""
        if self.llama is None and not is_llama_loaded():
            try:
                print("\n⏳ First low-confidence query detected!")
                print("Loading advanced model for better answers...\n")
                self.llama = get_llama_pipeline()
            except Exception as e:
                print(f"\n⚠️  Could not load LLaMA: {e}")
                print("Continuing with direct FAQ responses only.\n")
                return False
        elif is_llama_loaded():
            self.llama = get_llama_pipeline()
        
        return True
    
    def get_answer(self, user_query):
        """
        Main routing logic - decides between direct response and LLaMA
        
        Args:
            user_query: User's question
        
        Returns:
            tuple: (answer_text, route_taken, confidence)
        """
        self.stats["total_queries"] += 1
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"🔍 Query #{self.stats['total_queries']}: {user_query}")
            print(f"{'='*60}")
        
        # Get routing decision from retriever
        decision = self.retriever.get_best_match(user_query)
        
        # Handle no results
        if not decision['result']:
            self.stats["failed_queries"] += 1
            answer = self._generate_no_match_response()
            return answer, "no_match", "none"
        
        result = decision['result']
        confidence = result['confidence']
        route = decision['route']
        
        if self.debug:
            print(f"📊 Confidence: {confidence.upper()}")
            print(f"📍 Route: {route.upper()}")
            print(f"🎯 Score: {result['normalized_score']:.3f}")
        
        # FAST PATH: High confidence - direct response
        if route == "direct":
            self.stats["direct_responses"] += 1
            
            if self.debug:
                print("⚡ Using FAST PATH (direct response)")
            
            answer = format_response(result)
            return answer, "direct", confidence
        
        # SLOW PATH: Medium/Low confidence - use tree search + LLaMA
        else:
            self.stats["llama_responses"] += 1
            
            if self.debug:
                print("🤖 Using SLOW PATH (Tree Search + LLaMA synthesis)")
            
            # Load LLaMA if needed
            if not self._load_llama_if_needed():
                # Fallback to direct response if LLaMA fails
                if self.debug:
                    print("⚠️  Falling back to direct response")
                answer = format_response(result)
                return answer, "direct_fallback", confidence
            
            # Use tree search for better context (if enabled)
            from config import TREE_SEARCH_ENABLED, TREE_JSON_PATH
            
            if TREE_SEARCH_ENABLED:
                try:
                    # Lazy load tree searcher
                    if not hasattr(self, 'tree_searcher'):
                        from tree_search import TreeSearcher
                        import os
                        
                        if os.path.exists(TREE_JSON_PATH):
                            self.tree_searcher = TreeSearcher(
                                tree_path=TREE_JSON_PATH,
                                debug=self.debug
                            )
                            if self.debug:
                                print("🌲 Tree searcher loaded")
                        else:
                            self.tree_searcher = None
                            if self.debug:
                                print("⚠️  Tree not found, using fallback method")
                    
                    # Search tree for relevant FAQs
                    if self.tree_searcher:
                        if self.debug:
                            print("🔍 Searching tree for relevant documentation...")
                        
                        from config import TREE_SEARCH_MAX_NODES
                        relevant_faqs = self.tree_searcher.search(
                            user_query,
                            max_nodes=TREE_SEARCH_MAX_NODES
                        )
                        
                        if relevant_faqs:
                            context_faqs = relevant_faqs
                            if self.debug:
                                print(f"✅ Tree search found {len(context_faqs)} relevant FAQs")
                        else:
                            # Fallback to original top-3 method
                            context_faqs = decision.get('context_faqs', [result])
                            if self.debug:
                                print("⚠️  Tree search returned no results, using top-3 fallback")
                    else:
                        # Tree not available, use original method
                        context_faqs = decision.get('context_faqs', [result])
                
                except Exception as e:
                    if self.debug:
                        print(f"⚠️  Tree search failed: {e}")
                    # Fallback to original method
                    context_faqs = decision.get('context_faqs', [result])
            else:
                # Tree search disabled, use original method
                context_faqs = decision.get('context_faqs', [result])
                if self.debug:
                    print("ℹ️  Tree search disabled, using top-3 method")
            
            # Generate LLaMA response
            answer = self._generate_llama_answer(user_query, context_faqs, confidence)
            
            return answer, "llama", confidence
    
    def _generate_llama_answer(self, user_query, context_faqs, confidence):
        """
        Generate answer using LLaMA with appropriate prompt
        
        Args:
            user_query: User's question
            context_faqs: List of FAQ contexts
            confidence: Confidence level
        
        Returns:
            Generated answer string
        """
        context_faqs = self._normalize_context_faqs(context_faqs)

        # Build appropriate prompt
        if confidence == "high":
            prompt = build_high_confidence_prompt(user_query, context_faqs[0])
        elif confidence == "medium":
            prompt = build_medium_confidence_prompt(user_query, context_faqs)
        else:  # low or very_low
            prompt = build_low_confidence_prompt(user_query, context_faqs)
        
        # Generate response
        try:
            response = generate_llama_response(prompt, confidence_level=confidence)
            
            if response is None:
                raise Exception("LLaMA returned None")
            
            # Add source attribution
            top_result = context_faqs[0]
            
            if confidence == "medium":
                response += f"\n\n---\n📚 **Source:** {top_result['url']}"
                response += f"\n🏷️ **Category:** {top_result['category']}"
            elif confidence in ["low", "very_low"]:
                response += f"\n\n---"
                response += f"\n📚 **Related Documentation:** {top_result['url']}"
                response += f"\n🏷️ **Category:** {top_result['category']}"
                response += f"\n📧 **Need More Help?** {SUPPORT_LINK}"
                response += f"\n📖 **Browse Docs:** {ICER_DOCS_BASE}"
            
            return response
            
        except Exception as e:
            if self.debug:
                print(f"❌ LLaMA generation failed: {e}")
            
            # Fallback to formatted direct response
            return format_response(context_faqs[0])

    def _normalize_context_faqs(self, context_faqs):
        """
        Convert tree-search FAQ records into the retriever match schema used by
        prompt builders and response formatters.
        """
        normalized = []

        for faq in context_faqs:
            if "matched_question" in faq and "matched_answer" in faq:
                normalized.append(faq)
                continue

            normalized.append({
                "raw_score": float(faq.get("raw_score", 0.0)),
                "normalized_score": float(faq.get("normalized_score", 0.0)),
                "bi_score": float(faq.get("bi_score", 0.0)),
                "score_gap": float(faq.get("score_gap", 0.0)),
                "confidence": faq.get("confidence", "low"),
                "needs_llama": faq.get("needs_llama", True),
                "matched_question": faq.get("question", ""),
                "matched_answer": faq.get("answer", ""),
                "url": faq.get("url", "N/A"),
                "category": faq.get("category", "General"),
                "hash": faq.get("hash", "N/A")
            })

        return normalized
    
    def _generate_no_match_response(self):
        """Generate response when no matches found"""
        return f"""
❌ **No Matches Found**

I couldn't find relevant information in the ICER documentation for your question.

**🆘 Recommended Actions:**

1. **Submit a Support Ticket** (Best option for specific help)
   {SUPPORT_LINK}

2. **Browse Official Documentation**
   {ICER_DOCS_BASE}

3. **Refine Your Question**
   Try using different keywords or rephrasing your question.

**💡 Tips for Better Results:**
   • Use specific technical terms (e.g., "SLURM job submission" instead of "how to run code")
   • Mention the system/tool you're asking about (e.g., "GPU allocation", "Python environment")
   • Break complex questions into smaller parts
        """.strip()
    
    def print_stats(self):
        """Print usage statistics"""
        print(f"\n{'='*60}")
        print("📊 **Session Statistics**")
        print(f"{'='*60}")
        print(f"  Total Queries: {self.stats['total_queries']}")
        print(f"  Direct Responses (Fast): {self.stats['direct_responses']}")
        print(f"  LLaMA Responses (Slow): {self.stats['llama_responses']}")
        print(f"  Failed Queries: {self.stats['failed_queries']}")
        
        if self.stats['total_queries'] > 0:
            direct_pct = (self.stats['direct_responses'] / self.stats['total_queries']) * 100
            print(f"\n  ⚡ Speed Efficiency: {direct_pct:.1f}% queries used fast path")
        
        print(f"\n  🤖 LLaMA Status: {'Loaded ✅' if is_llama_loaded() else 'Not Loaded ❌'}")
        print(f"{'='*60}\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def run_cli():
    """
    Interactive CLI for the smart FAQ assistant
    """
    print("\n" + "="*60)
    print("💬 ICER Smart FAQ Assistant")
    print("="*60)
    print("\nWelcome! I'll help you find answers from ICER documentation.")
    print("I use FAST responses when confident, and SLOW (but accurate)")
    print("LLaMA synthesis for complex or ambiguous questions.\n")
    
    assistant = SmartFAQAssistant(debug=DEBUG_MODE)
    
    print("💡 **Commands:**")
    print("  • Type your question normally")
    print("  • 'stats' - View session statistics")
    print("  • 'help' - Show tips for better results")
    print("  • 'quit' or 'exit' - Exit the assistant")
    print("\n" + "="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Handle exit
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using ICER FAQ Assistant!")
                assistant.print_stats()
                print("Goodbye! 🎓\n")
                break
            
            # Handle stats command
            if user_input.lower() == "stats":
                assistant.print_stats()
                continue
            
            # Handle help command
            if user_input.lower() == "help":
                print(f"""
{'='*60}
💡 **Tips for Better Results**
{'='*60}

1. **Be Specific**
   ❌ "How do I run a job?"
   ✅ "How do I submit a SLURM job on the HPCC?"

2. **Mention the System/Tool**
   ❌ "Install packages"
   ✅ "How do I install Python packages with conda on ICER?"

3. **Use Technical Terms**
   ❌ "Make my code faster"
   ✅ "How do I request GPU resources in SLURM?"

4. **Break Complex Questions**
   ❌ "How do I set up everything for deep learning?"
   ✅ "How do I load CUDA modules?" (then ask next steps)

{'='*60}
                """)
                continue
            
            # Handle empty input
            if not user_input:
                continue
            
            # Get answer
            print()  # Blank line for readability
            answer, route, confidence = assistant.get_answer(user_input)
            
            # Show routing info in debug mode
            if DEBUG_MODE:
                route_emoji = "⚡" if route == "direct" else "🤖"
                print(f"\n[{route_emoji} Route: {route} | Confidence: {confidence}]")
            
            print(f"\nAI:\n{answer}\n")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("Type 'quit' to exit properly.\n")
            continue
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            print("Please try again or type 'quit' to exit.\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        run_cli()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
