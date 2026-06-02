# tree_search.py
"""
LLM-based tree search for FAQ retrieval
Implements reasoning-based navigation through hierarchical FAQ tree
"""

import json
from typing import List, Dict, Any, Optional
from prompt import get_llama_pipeline, generate_llama_response


class TreeSearcher:
    """
    Performs reasoning-based search through FAQ tree structure
    """
    
    def __init__(self, tree_path="faq_tree.json", debug=False):
        """
        Initialize tree searcher
        
        Args:
            tree_path: Path to tree JSON file
            debug: Enable debug output
        """
        self.tree_path = tree_path
        self.debug = debug
        self.tree = None
        self.llm = None
        self.load_tree()
    
    def load_tree(self):
        """Load tree structure from JSON"""
        if self.debug:
            print(f"Loading tree from {self.tree_path}...")
        
        with open(self.tree_path, 'r', encoding='utf-8') as f:
            self.tree = json.load(f)
        
        if self.debug:
            total_nodes = len(self.tree.get("nodes", []))
            total_faqs = self.tree.get("metadata", {}).get("total_faqs", 0)
            print(f"Loaded tree: {total_nodes} categories, {total_faqs} FAQs")
    
    def _ensure_llm_loaded(self):
        """Ensure LLaMA is loaded"""
        if self.llm is None:
            if self.debug:
                print("Loading LLaMA for tree search...")
            self.llm = get_llama_pipeline()
    
    def search(self, user_query: str, max_nodes=3) -> List[Dict]:
        """
        Search tree for relevant nodes using LLM reasoning
        
        Args:
            user_query: User's question
            max_nodes: Maximum number of nodes to return
            
        Returns:
            List of relevant FAQ dictionaries
        """
        if self.debug:
            print(f"\nTree Search for: '{user_query}'")
        
        self._ensure_llm_loaded()
        
        # Step 1: Find relevant top-level categories
        relevant_categories = self._search_top_level(user_query)
        
        if self.debug:
            print(f"  Relevant categories: {relevant_categories}")
        
        if not relevant_categories:
            if self.debug:
                print("  No relevant categories found")
            return []
        
        # Step 2: Search within relevant categories
        all_relevant_faqs = []
        
        for category_id in relevant_categories:
            category_node = self._find_node_by_id(category_id)
            
            if not category_node:
                continue
            
            if "subnodes" in category_node:
                # Has subcategories - search them
                relevant_subnodes = self._search_subnodes(
                    user_query,
                    category_node["subnodes"],
                    category_node["title"]
                )
                
                if self.debug:
                    print(f"    Relevant subnodes in '{category_node['title']}': {relevant_subnodes}")
                
                # Collect FAQs from relevant subnodes
                for subnode_id in relevant_subnodes:
                    subnode = self._find_subnode_by_id(category_node["subnodes"], subnode_id)
                    if subnode:
                        all_relevant_faqs.extend(subnode.get("faqs", []))
            else:
                # No subcategories - get FAQs directly
                all_relevant_faqs.extend(category_node.get("faqs", []))
        
        if self.debug:
            print(f"  Total FAQs retrieved: {len(all_relevant_faqs)}")
        
        # Step 3: If too many FAQs, filter further with LLM
        if len(all_relevant_faqs) > 10:
            all_relevant_faqs = self._filter_faqs(user_query, all_relevant_faqs)
            
            if self.debug:
                print(f"  After filtering: {len(all_relevant_faqs)} FAQs")
        
        return all_relevant_faqs[:max_nodes * 5]  # Return reasonable number
    
    def _search_top_level(self, user_query: str) -> List[str]:
        """
        Search top-level categories
        
        Args:
            user_query: User's question
            
        Returns:
            List of relevant node IDs
        """
        # Build simplified tree structure for LLM
        simplified_tree = []
        for node in self.tree.get("nodes", []):
            simplified_tree.append({
                "node_id": node["node_id"],
                "title": node["title"],
                "summary": node["summary"],
                "faq_count": node.get("faq_count", 0)
            })
        
        prompt = f"""You are searching a knowledge base to answer a user's question.

User Question: {user_query}

Available categories in the knowledge base:
{json.dumps(simplified_tree, indent=2)}

Which categories are most likely to contain information relevant to this question?
Select 1-3 categories that seem most relevant.

Return ONLY a JSON object:
{{
  "thinking": "brief reasoning about relevance",
  "node_ids": ["node_id1", "node_id2"]
}}

Response:"""
        
        response = generate_llama_response(prompt, confidence_level="high")
        
        try:
            # Extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                node_ids = result.get("node_ids", [])
                
                if self.debug and "thinking" in result:
                    print(f"  LLM reasoning: {result['thinking']}")
                
                return node_ids
        except Exception as e:
            if self.debug:
                print(f"  Error parsing LLM response: {e}")
        
        return []
    
    def _search_subnodes(self, user_query: str, subnodes: List[Dict], parent_title: str) -> List[str]:
        """
        Search subnodes within a category
        
        Args:
            user_query: User's question
            subnodes: List of subnode dictionaries
            parent_title: Title of parent category
            
        Returns:
            List of relevant subnode IDs
        """
        # Build simplified subnode structure
        simplified_subnodes = []
        for subnode in subnodes:
            simplified_subnodes.append({
                "node_id": subnode["node_id"],
                "title": subnode["title"],
                "summary": subnode["summary"],
                "faq_count": subnode.get("faq_count", 0)
            })
        
        prompt = f"""You are searching within the "{parent_title}" category to answer a user's question.

User Question: {user_query}

Available subcategories:
{json.dumps(simplified_subnodes, indent=2)}

Which subcategories are most likely to contain the answer?
Select 1-2 most relevant subcategories.

Return ONLY a JSON object:
{{
  "thinking": "brief reasoning",
  "node_ids": ["node_id1", "node_id2"]
}}

Response:"""
        
        response = generate_llama_response(prompt, confidence_level="high")
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                return result.get("node_ids", [])
        except:
            pass
        
        # Fallback: return all subnodes
        return [sn["node_id"] for sn in subnodes]
    
    def _filter_faqs(self, user_query: str, faqs: List[Dict], keep_top=10) -> List[Dict]:
        """
        Filter FAQs to most relevant ones using LLM
        
        Args:
            user_query: User's question
            faqs: List of FAQ dictionaries
            keep_top: Number of FAQs to keep
            
        Returns:
            Filtered list of FAQs
        """
        # Create list of questions only
        questions = [{"index": i, "question": faq["question"]} for i, faq in enumerate(faqs)]
        
        prompt = f"""You are filtering FAQs to find the most relevant ones for a user's question.

User Question: {user_query}

Available FAQs:
{json.dumps(questions, indent=2)}

Which FAQ questions are most relevant to answering the user's question?
Return the indices of the top {keep_top} most relevant FAQs.

Return ONLY a JSON object:
{{
  "thinking": "brief reasoning",
  "relevant_indices": [0, 3, 5, ...]
}}

Response:"""
        
        response = generate_llama_response(prompt, confidence_level="high")
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                relevant_indices = result.get("relevant_indices", [])
                
                # Return FAQs at those indices
                filtered = [faqs[i] for i in relevant_indices if i < len(faqs)]
                return filtered
        except:
            pass
        
        # Fallback: return first N
        return faqs[:keep_top]
    
    def _find_node_by_id(self, node_id: str) -> Optional[Dict]:
        """Find node by ID in tree"""
        for node in self.tree.get("nodes", []):
            if node["node_id"] == node_id:
                return node
        return None
    
    def _find_subnode_by_id(self, subnodes: List[Dict], node_id: str) -> Optional[Dict]:
        """Find subnode by ID"""
        for subnode in subnodes:
            if subnode["node_id"] == node_id:
                return subnode
        return None
    
    def get_tree_summary(self) -> str:
        """
        Get human-readable tree summary
        
        Returns:
            Summary string
        """
        if not self.tree:
            return "No tree loaded"
        
        summary = []
        summary.append(f"Tree Structure ({self.tree['metadata']['total_faqs']} total FAQs):")
        summary.append("")
        
        for node in self.tree.get("nodes", []):
            summary.append(f"  {node['title']} ({node.get('faq_count', 0)} FAQs)")
            summary.append(f"    {node['summary']}")
            
            if "subnodes" in node:
                for subnode in node["subnodes"]:
                    summary.append(f"      - {subnode['title']} ({subnode.get('faq_count', 0)} FAQs)")
            
            summary.append("")
        
        return "\n".join(summary)


def main():
    """Test tree search from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search FAQ tree")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--tree-path", default="faq_tree.json", help="Path to tree JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--max-nodes", type=int, default=3, help="Max nodes to return")
    
    args = parser.parse_args()
    
    searcher = TreeSearcher(tree_path=args.tree_path, debug=args.debug)
    
    # Show tree summary
    if args.debug:
        print(searcher.get_tree_summary())
        print("\n" + "="*60)
    
    # Perform search
    results = searcher.search(args.query, max_nodes=args.max_nodes)
    
    print(f"\nFound {len(results)} relevant FAQs:")
    print("="*60)
    
    for i, faq in enumerate(results, 1):
        print(f"\n{i}. Q: {faq['question']}")
        print(f"   A: {faq['answer'][:150]}...")
        print(f"   Category: {faq.get('category', 'N/A')}")
        print(f"   URL: {faq.get('url', 'N/A')}")


if __name__ == "__main__":
    main()
