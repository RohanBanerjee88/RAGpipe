# tree_builder.py
"""
Automatic hierarchical tree builder for FAQ documents
Builds a PageIndex-style tree structure without manual categorization
"""

import json
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any
from prompt import get_llama_pipeline, generate_llama_response


class FAQTreeBuilder:
    """
    Builds hierarchical tree structure from flat FAQ list
    Uses LLM to automatically categorize and organize FAQs
    """
    
    def __init__(self, faq_json_path="all_faqs.json", tree_output_path="faq_tree.json", debug=False):
        """
        Initialize tree builder
        
        Args:
            faq_json_path: Path to FAQ JSON file
            tree_output_path: Path to save generated tree
            debug: Enable debug output
        """
        self.faq_json_path = faq_json_path
        self.tree_output_path = tree_output_path
        self.debug = debug
        self.faqs = []
        self.tree = None
        self.llm = None
        
    def load_faqs(self):
        """Load FAQs from JSON file"""
        if self.debug:
            print(f"Loading FAQs from {self.faq_json_path}...")
        
        with open(self.faq_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both old and new formats
        if "faqs" in data:
            self.faqs = data["faqs"]
        else:
            self.faqs = data
        
        if self.debug:
            print(f"Loaded {len(self.faqs)} FAQs")
    
    def _ensure_llm_loaded(self):
        """Ensure LLaMA is loaded for tree building"""
        if self.llm is None:
            if self.debug:
                print("Loading LLaMA for tree building...")
            self.llm = get_llama_pipeline()
    
    def _generate_node_id(self, level, index):
        """Generate unique node ID"""
        return f"{level:04d}_{index:04d}"
    
    def build_initial_categories(self):
        """
        Build top-level categories from existing FAQ categories
        Uses LLM to consolidate similar categories
        
        Returns:
            dict: Category mapping
        """
        if self.debug:
            print("\nStep 1: Building initial categories...")
        
        # Collect unique categories from FAQs
        existing_categories = defaultdict(list)
        for faq in self.faqs:
            category = faq.get('category', 'General')
            existing_categories[category].append(faq)
        
        if self.debug:
            print(f"Found {len(existing_categories)} existing categories:")
            for cat, faqs in existing_categories.items():
                print(f"  - {cat}: {len(faqs)} FAQs")
        
        # If too many categories, consolidate with LLM
        if len(existing_categories) > 10:
            if self.debug:
                print("\nConsolidating categories with LLM...")
            
            consolidated = self._consolidate_categories(list(existing_categories.keys()))
            
            # Remap FAQs to consolidated categories
            category_mapping = {}
            for old_cat in existing_categories.keys():
                category_mapping[old_cat] = self._find_best_match(old_cat, consolidated)
            
            # Rebuild with consolidated categories
            new_categories = defaultdict(list)
            for faq in self.faqs:
                old_cat = faq.get('category', 'General')
                new_cat = category_mapping.get(old_cat, 'General')
                new_categories[new_cat].append(faq)
            
            existing_categories = new_categories
            
            if self.debug:
                print(f"Consolidated to {len(existing_categories)} categories")
        
        return existing_categories
    
    def _consolidate_categories(self, categories: List[str]) -> List[str]:
        """
        Use LLM to consolidate similar categories
        
        Args:
            categories: List of category names
            
        Returns:
            List of consolidated category names
        """
        self._ensure_llm_loaded()
        
        prompt = f"""You are organizing FAQ categories for an HPC/computing support system.

Given categories:
{json.dumps(categories, indent=2)}

Consolidate these into 5-8 high-level categories that make sense.
Combine similar categories (e.g., "Login Issues" and "Account Access" → "Account Management").

Return ONLY a JSON array of consolidated category names:
["Category 1", "Category 2", ...]

Response:"""
        
        response = generate_llama_response(prompt, confidence_level="high")
        
        try:
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                consolidated = json.loads(response[start:end])
                return consolidated
        except:
            pass
        
        # Fallback: return top 8 categories by FAQ count
        return categories[:8]
    
    def _find_best_match(self, category: str, consolidated: List[str]) -> str:
        """Find best matching consolidated category"""
        # Simple keyword matching (could use LLM for better results)
        category_lower = category.lower()
        
        for consol in consolidated:
            if category_lower in consol.lower() or consol.lower() in category_lower:
                return consol
        
        # Return first consolidated category as fallback
        return consolidated[0] if consolidated else "General"
    
    def build_subcategories(self, category_name: str, faqs: List[Dict]) -> List[str]:
        """
        Build subcategories for a category using LLM
        
        Args:
            category_name: Name of parent category
            faqs: FAQs in this category
            
        Returns:
            List of subcategory names or None
        """
        if self.debug:
            print(f"\n  Building subcategories for '{category_name}' ({len(faqs)} FAQs)...")
        
        # If few FAQs, don't create subcategories
        if len(faqs) <= 5:
            if self.debug:
                print(f"    Skipping (too few FAQs)")
            return None
        
        self._ensure_llm_loaded()
        
        # Get question summaries
        questions = [faq['question'] for faq in faqs[:20]]  # Sample first 20
        
        prompt = f"""You are organizing FAQs for the category: "{category_name}"

Sample questions:
{json.dumps(questions, indent=2)}

Create 2-4 subcategories that logically group these questions.
Each subcategory should represent a distinct topic.

Return ONLY a JSON array of subcategory names:
["Subcategory 1", "Subcategory 2", ...]

Response:"""
        
        response = generate_llama_response(prompt, confidence_level="high")
        
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                subcategories = json.loads(response[start:end])
                
                if self.debug:
                    print(f"    Created subcategories: {subcategories}")
                
                return subcategories
        except:
            pass
        
        return None
    
    def assign_faqs_to_subcategories(self, faqs: List[Dict], subcategories: List[str]) -> Dict:
        """
        Assign FAQs to subcategories using LLM
        
        Args:
            faqs: List of FAQs
            subcategories: List of subcategory names
            
        Returns:
            dict: Mapping of subcategory -> FAQs
        """
        self._ensure_llm_loaded()
        
        # BUG FIX: Flatten subcategories if LLM returned nested list
        flat_subcategories = []
        for subcat in subcategories:
            if isinstance(subcat, list):
                # LLM returned nested list, flatten it
                flat_subcategories.extend(subcat)
            else:
                flat_subcategories.append(subcat)
        
        subcategories = flat_subcategories
        
        # Now safe to create dict (all elements are strings, not lists)
        subcategory_map = {subcat: [] for subcat in subcategories}
        subcategory_map["Other"] = []  # Catch-all
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(faqs), batch_size):
            batch = faqs[i:i+batch_size]
            
            questions = [faq['question'] for faq in batch]
            
            prompt = f"""Assign each question to the most appropriate subcategory.

Subcategories:
{json.dumps(subcategories, indent=2)}

Questions:
{json.dumps(questions, indent=2)}

Return ONLY a JSON array where each element is the subcategory name for the corresponding question.
If none fit, use "Other".

Example: ["Subcategory 1", "Subcategory 2", "Other", ...]

Response:"""
            
            response = generate_llama_response(prompt, confidence_level="high")
            
            try:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    assignments = json.loads(response[start:end])
                    
                    for faq, assignment in zip(batch, assignments):
                        if assignment in subcategory_map:
                            subcategory_map[assignment].append(faq)
                        else:
                            subcategory_map["Other"].append(faq)
            except:
                # Fallback: assign to "Other"
                subcategory_map["Other"].extend(batch)
        
        # Remove empty subcategories
        subcategory_map = {k: v for k, v in subcategory_map.items() if v}
        
        return subcategory_map
    
    def generate_summary(self, category_name: str, faqs: List[Dict], max_faqs=5) -> str:
        """
        Generate summary for a category/node
        
        Args:
            category_name: Name of category
            faqs: FAQs in this category
            max_faqs: Number of FAQs to sample for summary
            
        Returns:
            str: Summary text
        """
        sample_questions = [faq['question'] for faq in faqs[:max_faqs]]
        
        self._ensure_llm_loaded()
        
        prompt = f"""Summarize what topics are covered in the category "{category_name}".

Sample questions from this category:
{json.dumps(sample_questions, indent=2)}

Write a one-sentence summary (max 100 characters) describing what this category covers.

Response:"""
        
        summary = generate_llama_response(prompt, confidence_level="high")
        
        # Clean up summary
        summary = summary.strip().split('\n')[0]  # First line only
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
    
    def build_tree(self):
        """
        Build complete hierarchical tree structure
        
        Returns:
            dict: Tree structure
        """
        print("\n" + "="*60)
        print("Building FAQ Tree Structure")
        print("="*60)
        
        # Step 1: Load FAQs
        self.load_faqs()
        
        # Step 2: Build top-level categories
        categories = self.build_initial_categories()
        
        # Step 3: Build hierarchical tree
        tree = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_faqs": len(self.faqs),
                "total_categories": len(categories),
                "version": "1.0"
            },
            "nodes": []
        }
        
        node_index = 0
        for category_name, category_faqs in categories.items():
            if self.debug:
                print(f"\nProcessing category: {category_name} ({len(category_faqs)} FAQs)")
            
            # Generate category summary
            category_summary = self.generate_summary(category_name, category_faqs)
            
            # Build category node
            category_node = {
                "node_id": self._generate_node_id(1, node_index),
                "title": category_name,
                "summary": category_summary,
                "faq_count": len(category_faqs),
                "faqs": []
            }
            
            # Try to build subcategories
            subcategories = self.build_subcategories(category_name, category_faqs)
            
            if subcategories:
                # Has subcategories
                subcat_map = self.assign_faqs_to_subcategories(category_faqs, subcategories)
                
                category_node["subnodes"] = []
                
                subnode_index = 0
                for subcat_name, subcat_faqs in subcat_map.items():
                    subcat_summary = self.generate_summary(subcat_name, subcat_faqs)
                    
                    subnode = {
                        "node_id": self._generate_node_id(2, subnode_index),
                        "title": subcat_name,
                        "summary": subcat_summary,
                        "faq_count": len(subcat_faqs),
                        "faqs": subcat_faqs
                    }
                    
                    category_node["subnodes"].append(subnode)
                    subnode_index += 1
            else:
                # No subcategories, store FAQs directly
                category_node["faqs"] = category_faqs
            
            tree["nodes"].append(category_node)
            node_index += 1
        
        self.tree = tree
        
        print("\n" + "="*60)
        print("Tree Building Complete")
        print("="*60)
        print(f"Total nodes: {len(tree['nodes'])}")
        total_subnodes = sum(len(node.get('subnodes', [])) for node in tree['nodes'])
        print(f"Total subnodes: {total_subnodes}")
        print(f"Total FAQs: {tree['metadata']['total_faqs']}")
        print("="*60 + "\n")
        
        return tree
    
    def save_tree(self):
        """Save tree to JSON file"""
        if self.tree is None:
            raise ValueError("No tree built yet. Call build_tree() first.")
        
        with open(self.tree_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.tree, f, indent=2, ensure_ascii=False)
        
        print(f"Tree saved to {self.tree_output_path}")
    
    def update_tree_incremental(self, new_faqs: List[Dict]):
        """
        Update existing tree with new FAQs (incremental update)
        
        Args:
            new_faqs: List of new FAQ dictionaries
        """
        if self.debug:
            print(f"\nIncremental update: Adding {len(new_faqs)} new FAQs...")
        
        # Load existing tree
        try:
            with open(self.tree_output_path, 'r', encoding='utf-8') as f:
                self.tree = json.load(f)
        except FileNotFoundError:
            if self.debug:
                print("No existing tree found. Building from scratch...")
            return self.build_tree()
        
        # For each new FAQ, find best matching node
        self._ensure_llm_loaded()
        
        for faq in new_faqs:
            best_node = self._find_best_node_for_faq(faq)
            
            if best_node:
                best_node["faqs"].append(faq)
                best_node["faq_count"] = len(best_node["faqs"])
            else:
                # Create new category if no good match
                if self.debug:
                    print(f"  Creating new category for: {faq['question'][:50]}...")
                # Add to "Other" or create new node
                self._add_to_other_node(faq)
        
        # Update metadata
        self.tree["metadata"]["updated_at"] = datetime.now().isoformat()
        self.tree["metadata"]["total_faqs"] = sum(
            self._count_faqs_in_node(node) for node in self.tree["nodes"]
        )
        
        if self.debug:
            print(f"Tree updated. Total FAQs: {self.tree['metadata']['total_faqs']}")
    
    def _find_best_node_for_faq(self, faq: Dict) -> Dict:
        """Find best matching node for a new FAQ"""
        # Simple approach: use existing category if matches
        faq_category = faq.get('category', '')
        
        for node in self.tree["nodes"]:
            if node["title"].lower() == faq_category.lower():
                # Check subnodes
                if "subnodes" in node:
                    for subnode in node["subnodes"]:
                        # Simple keyword matching (could use LLM)
                        return subnode
                return node
        
        return None
    
    def _add_to_other_node(self, faq: Dict):
        """Add FAQ to 'Other' category or create it"""
        # Find or create "Other" node
        other_node = None
        for node in self.tree["nodes"]:
            if node["title"] == "Other":
                other_node = node
                break
        
        if not other_node:
            other_node = {
                "node_id": self._generate_node_id(1, len(self.tree["nodes"])),
                "title": "Other",
                "summary": "Miscellaneous topics",
                "faq_count": 0,
                "faqs": []
            }
            self.tree["nodes"].append(other_node)
        
        other_node["faqs"].append(faq)
        other_node["faq_count"] = len(other_node["faqs"])
    
    def _count_faqs_in_node(self, node: Dict) -> int:
        """Count total FAQs in a node and its subnodes"""
        count = len(node.get("faqs", []))
        if "subnodes" in node:
            for subnode in node["subnodes"]:
                count += len(subnode.get("faqs", []))
        return count


def main():
    """Build tree from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAQ tree structure")
    parser.add_argument("--faq-path", default="all_faqs.json", help="Path to FAQ JSON")
    parser.add_argument("--output", default="faq_tree.json", help="Output tree path")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    builder = FAQTreeBuilder(
        faq_json_path=args.faq_path,
        tree_output_path=args.output,
        debug=args.debug
    )
    
    tree = builder.build_tree()
    builder.save_tree()
    
    print("\nTree structure preview:")
    for node in tree["nodes"][:3]:  # Show first 3 categories
        print(f"\n{node['title']} ({node['faq_count']} FAQs)")
        print(f"  Summary: {node['summary']}")
        if "subnodes" in node:
            for subnode in node["subnodes"]:
                print(f"    - {subnode['title']} ({subnode['faq_count']} FAQs)")


if __name__ == "__main__":
    main()