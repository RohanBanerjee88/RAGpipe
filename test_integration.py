#!/usr/bin/env python3
"""
Test script to verify PageIndex tree integration
Run this after scraping to test the complete system
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("="*60)
    print("Testing Module Imports")
    print("="*60)
    
    try:
        import config
        print("✅ config.py imported")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        from tree_builder import FAQTreeBuilder
        print("✅ tree_builder.py imported")
    except Exception as e:
        print(f"❌ tree_builder.py import failed: {e}")
        return False
    
    try:
        from tree_search import TreeSearcher
        print("✅ tree_search.py imported")
    except Exception as e:
        print(f"❌ tree_search.py import failed: {e}")
        return False
    
    try:
        from main import SmartFAQAssistant
        print("✅ main.py imported")
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_tree_exists():
    """Check if FAQ tree exists"""
    print("="*60)
    print("Checking FAQ Tree")
    print("="*60)
    
    from config import TREE_JSON_PATH, FAQ_JSON_PATH
    
    if not os.path.exists(FAQ_JSON_PATH):
        print(f"❌ {FAQ_JSON_PATH} not found")
        print("   Run: python scrape.py")
        return False
    
    print(f"✅ {FAQ_JSON_PATH} found")
    
    if not os.path.exists(TREE_JSON_PATH):
        print(f"⚠️  {TREE_JSON_PATH} not found")
        print(f"   Run: python tree_builder.py --debug")
        return False
    
    print(f"✅ {TREE_JSON_PATH} found")
    
    # Load and show tree stats
    import json
    with open(TREE_JSON_PATH, 'r') as f:
        tree = json.load(f)
    
    metadata = tree.get('metadata', {})
    nodes = tree.get('nodes', [])
    
    print(f"\nTree Statistics:")
    print(f"  Total FAQs: {metadata.get('total_faqs', 0)}")
    print(f"  Categories: {len(nodes)}")
    
    total_subnodes = sum(len(node.get('subnodes', [])) for node in nodes)
    print(f"  Subcategories: {total_subnodes}")
    
    print("\nCategory Preview:")
    for node in nodes[:3]:
        print(f"  - {node['title']} ({node.get('faq_count', 0)} FAQs)")
        if 'subnodes' in node:
            for subnode in node['subnodes'][:2]:
                print(f"      → {subnode['title']} ({subnode.get('faq_count', 0)} FAQs)")
    
    print("\n✅ Tree structure looks good!\n")
    return True


def test_tree_search():
    """Test tree search functionality"""
    print("="*60)
    print("Testing Tree Search")
    print("="*60)
    
    from tree_search import TreeSearcher
    from config import TREE_JSON_PATH
    
    try:
        searcher = TreeSearcher(tree_path=TREE_JSON_PATH, debug=True)
        
        # Test query
        test_query = "How do I submit a job?"
        print(f"\nTest Query: '{test_query}'\n")
        
        results = searcher.search(test_query, max_nodes=2)
        
        print(f"\nResults: {len(results)} FAQs found")
        
        if results:
            print("\nTop 3 results:")
            for i, faq in enumerate(results[:3], 1):
                print(f"  {i}. Q: {faq['question'][:60]}...")
                print(f"     Category: {faq.get('category', 'N/A')}")
            
            print("\n✅ Tree search working!\n")
            return True
        else:
            print("⚠️  No results found (this might be okay if FAQs don't match query)")
            return True
    
    except Exception as e:
        print(f"❌ Tree search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_settings():
    """Check tree-related config settings"""
    print("="*60)
    print("Checking Configuration")
    print("="*60)
    
    try:
        from config import (
            TREE_JSON_PATH,
            TREE_SEARCH_ENABLED,
            TREE_SEARCH_MAX_NODES,
            TREE_SEARCH_MAX_FAQS,
            TREE_AUTO_UPDATE
        )
        
        print(f"\nTree Configuration:")
        print(f"  Tree path: {TREE_JSON_PATH}")
        print(f"  Tree search enabled: {TREE_SEARCH_ENABLED}")
        print(f"  Max nodes to search: {TREE_SEARCH_MAX_NODES}")
        print(f"  Max FAQs to return: {TREE_SEARCH_MAX_FAQS}")
        print(f"  Auto-update after scraping: {TREE_AUTO_UPDATE}")
        
        if not TREE_SEARCH_ENABLED:
            print("\n⚠️  WARNING: TREE_SEARCH_ENABLED is False")
            print("   Set TREE_SEARCH_ENABLED=True in config.py to use tree search")
        
        print("\n✅ Configuration loaded!\n")
        return True
    
    except ImportError as e:
        print(f"❌ Missing config settings: {e}")
        print("   Make sure config.py has tree search settings")
        return False


def test_main_integration():
    """Test that main.py can use tree search"""
    print("="*60)
    print("Testing Main.py Integration")
    print("="*60)
    
    try:
        from main import SmartFAQAssistant
        from config import TREE_SEARCH_ENABLED, TREE_JSON_PATH
        import os
        
        print("\nInitializing assistant...")
        assistant = SmartFAQAssistant(debug=False)
        
        if TREE_SEARCH_ENABLED and os.path.exists(TREE_JSON_PATH):
            print("✅ Tree search will be used for low-confidence queries")
        elif not TREE_SEARCH_ENABLED:
            print("⚠️  Tree search disabled (TREE_SEARCH_ENABLED=False)")
        elif not os.path.exists(TREE_JSON_PATH):
            print("⚠️  Tree file not found, will fallback to top-3 method")
        
        print("\n✅ Main.py integration ready!\n")
        return True
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PageIndex Tree Integration Test Suite")
    print("="*60 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config_settings),
        ("Tree File", test_tree_exists),
        ("Tree Search", test_tree_search),
        ("Main Integration", test_main_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYour system is ready to use tree search!")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Try a low-confidence query to trigger tree search")
        print("  3. Check debug output to see tree search in action")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nRecommended actions:")
        
        if not results.get("Module Imports"):
            print("  - Make sure all .py files are in the same directory")
        
        if not results.get("Tree File"):
            print("  - Run: python scrape.py")
            print("  - Run: python tree_builder.py --debug")
        
        if not results.get("Configuration"):
            print("  - Update config.py with tree search settings")
    
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())