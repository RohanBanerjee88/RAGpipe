# scrape.py
"""
Simple, reliable FAQ scraper with minimal filtering
Based on original scraper with deduplication added
"""

import requests
from bs4 import BeautifulSoup
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Import from config
try:
    from config import (
        SCRAPE_URLS,
        SCRAPE_TIMEOUT,
        SCRAPE_MAX_WORKERS,
        MIN_QUESTION_LENGTH,
        MIN_ANSWER_LENGTH,
        FAQ_JSON_PATH,
        SCRAPE_METADATA_PATH
    )
except ImportError:
    # Fallback if config.py not found
    print("⚠️  config.py not found, using default values")
    SCRAPE_URLS = [
        "https://docs.icer.msu.edu/Frequently_Asked_Questions_FAQ_/",
        "https://docs.icer.msu.edu/Linux_Shell/",
        "https://docs.icer.msu.edu/Submitting_a_Help_Ticket/",
        "https://docs.icer.msu.edu/Sensitive_Data_on_the_HPCC/"
    ]
    SCRAPE_TIMEOUT = 10
    SCRAPE_MAX_WORKERS = 5
    MIN_QUESTION_LENGTH = 10
    MIN_ANSWER_LENGTH = 20
    FAQ_JSON_PATH = "all_faqs.json"
    SCRAPE_METADATA_PATH = "scrape_metadata.json"


def generate_hash(question, answer):
    """Generate unique hash for Q&A pair to detect duplicates"""
    content = f"{question.lower().strip()}{answer.lower().strip()}"
    return hashlib.md5(content.encode()).hexdigest()


def extract_faqs_from_url(url, seen_hashes):
    """
    Extract FAQs from a single URL with minimal filtering
    Very similar to original scraper
    """
    try:
        print(f"📡 Fetching {url}...")
        response = requests.get(url, timeout=SCRAPE_TIMEOUT)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return [], 0

    soup = BeautifulSoup(response.text, "html.parser")
    faqs = []
    duplicates = 0
    
    # Find all potential question headings (h2, h3, h4)
    question_tags = soup.find_all(['h2', 'h3', 'h4'])
    current_section = "General"

    for tag in question_tags:
        question = tag.get_text(strip=True)

        # Very basic length check only
        if not question or len(question) < MIN_QUESTION_LENGTH:
            continue

        # Update category if it's a top-level heading (h2)
        if tag.name == "h2":
            current_section = question

        # Extract answer: everything until next heading
        answer_parts = []
        next_element = tag.find_next_sibling()
        
        while next_element and next_element.name not in ['h2', 'h3', 'h4']:
            if next_element.name in ['p', 'ul', 'ol', 'div']:
                text = next_element.get_text(strip=True)
                if text:
                    answer_parts.append(text)
            next_element = next_element.find_next_sibling()

        answer = " ".join(answer_parts).strip()
        
        # Basic answer length check
        if not answer or len(answer) < MIN_ANSWER_LENGTH:
            continue
        
        # Check for duplicates using hash
        content_hash = generate_hash(question, answer)
        if content_hash in seen_hashes:
            duplicates += 1
            continue
        
        seen_hashes.add(content_hash)
        
        # Add to FAQs
        faqs.append({
            "url": url,
            "category": current_section,
            "question": question,
            "answer": answer,
            "hash": content_hash,
            "scraped_at": datetime.now().isoformat()
        })

    print(f"✅ Scraped {len(faqs)} unique FAQs from {url} (removed {duplicates} duplicates)")
    return faqs, duplicates


def main():
    """Run the scraper"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting FAQ scrape from {len(SCRAPE_URLS)} URLs...")
    print(f"{'='*60}\n")
    
    all_faqs = []
    seen_hashes = set()
    total_duplicates = 0
    
    # Scrape all URLs concurrently
    with ThreadPoolExecutor(max_workers=SCRAPE_MAX_WORKERS) as executor:
        future_to_url = {
            executor.submit(extract_faqs_from_url, url, seen_hashes): url 
            for url in SCRAPE_URLS
        }
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                faqs, duplicates = future.result()
                all_faqs.extend(faqs)
                total_duplicates += duplicates
            except Exception as exc:
                print(f"❌ Error scraping {url}: {exc}")

    # Save to JSON
    data = {
        "metadata": {
            "last_scrape": datetime.now().isoformat(),
            "total_faqs": len(all_faqs),
            "urls_scraped": SCRAPE_URLS,
            "duplicates_removed": total_duplicates
        },
        "faqs": all_faqs
    }
    
    with open(FAQ_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✨ Scraping complete!")
    print(f"{'='*60}")
    print(f"📚 Total FAQs collected: {len(all_faqs)}")
    print(f"🔄 Duplicates removed: {total_duplicates}")
    print(f"💾 Saved to: {FAQ_JSON_PATH}")
    print(f"{'='*60}\n")
    
    # Also save metadata
    with open(SCRAPE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data["metadata"], f, indent=2)
    
    # Show sample
    if all_faqs:
        print("📋 Sample FAQ:")
        sample = all_faqs[0]
        print(f"   Q: {sample['question'][:80]}...")
        print(f"   A: {sample['answer'][:100]}...")
        print(f"   Category: {sample['category']}")
        print(f"   URL: {sample['url']}")
        print()
    
    # Update FAQ tree (if enabled)
    try:
        from config import TREE_AUTO_UPDATE, TREE_JSON_PATH
        
        if TREE_AUTO_UPDATE:
            print("🌲 Updating FAQ tree...")
            
            from tree_builder import FAQTreeBuilder
            import os
            
            builder = FAQTreeBuilder(
                faq_json_path=FAQ_JSON_PATH,
                tree_output_path=TREE_JSON_PATH,
                debug=False
            )
            
            # Check if tree exists
            if os.path.exists(TREE_JSON_PATH):
                print("   Using incremental update...")
                builder.update_tree_incremental(all_faqs)
            else:
                print("   Building tree from scratch...")
                builder.build_tree()
            
            builder.save_tree()
            print("✅ Tree updated successfully\n")
        else:
            print("ℹ️  Tree auto-update disabled (set TREE_AUTO_UPDATE=True in config.py to enable)\n")
    
    except ImportError:
        print("⚠️  tree_builder.py not found - tree not updated")
        print("   Run 'python tree_builder.py' manually to build tree\n")
    except Exception as e:
        print(f"⚠️  Tree update failed: {e}")
        print("   Run 'python tree_builder.py' manually to rebuild tree\n")


if __name__ == "__main__":
    main()