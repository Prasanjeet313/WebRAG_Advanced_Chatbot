"""
Selenium-Based Webpage Extractor with JSON Database
Extracts webpage content bypassing Cloudflare and saves to a shared JSON cache
This cache can be uploaded to GitHub for use by Streamlit deployment
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CACHE_FILE = "webpage_cache.json"
CHROME_VERSION = 144  # Update this to match your Chrome version


class WebpageCacheDB:
    """Manages the JSON-based webpage cache database"""
    
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load existing cache from JSON file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded cache with {len(data)} entries from {self.cache_file}")
                    return data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return {}
        else:
            logger.info(f"Cache file {self.cache_file} not found, creating new cache")
            return {}
    
    def _save_cache(self):
        """Save cache to JSON file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved cache with {len(self.cache_data)} entries to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def add_entry(self, url: str, title: str, content: str, metadata: Optional[Dict] = None):
        """Add or update a webpage entry in the cache"""
        entry = {
            'url': url,
            'title': title,
            'content': content,
            'timestamp': datetime.utcnow().isoformat(),
            'extracted_with': 'selenium',
            'content_length': len(content),
            'metadata': metadata or {}
        }
        
        self.cache_data[url] = entry
        self._save_cache()
        logger.info(f"Added/Updated cache entry for: {url}")
    
    def get_entry(self, url: str) -> Optional[Dict[str, Any]]:
        """Get a webpage entry from cache"""
        return self.cache_data.get(url)
    
    def has_entry(self, url: str) -> bool:
        """Check if URL exists in cache"""
        return url in self.cache_data
    
    def list_urls(self) -> List[str]:
        """List all cached URLs"""
        return list(self.cache_data.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_data:
            return {
                'total_entries': 0,
                'total_content_chars': 0,
                'urls': []
            }
        
        return {
            'total_entries': len(self.cache_data),
            'total_content_chars': sum(e.get('content_length', 0) for e in self.cache_data.values()),
            'urls': list(self.cache_data.keys()),
            'cache_file': str(self.cache_file)
        }


class SeleniumWebScraper:
    """Enhanced Selenium scraper with better Cloudflare bypass"""
    
    def __init__(self, headless: bool = True, chrome_version: int = CHROME_VERSION):
        self.headless = headless
        self.chrome_version = chrome_version
        self.driver = None
    
    def _init_driver(self):
        """Initialize undetected ChromeDriver"""
        if self.driver:
            return
        
        logger.info(f"Initializing undetected Chrome WebDriver (version {self.chrome_version})...")
        try:
            options = uc.ChromeOptions()
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            
            if self.headless:
                options.add_argument('--headless=new')
            
            self.driver = uc.Chrome(options=options, version_main=self.chrome_version)
            logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def scrape_url(self, url: str, wait_time: int = 10, max_retries: int = 3) -> Dict[str, Any]:
        """
        Scrape content from URL with retry logic
        
        Args:
            url: URL to scrape
            wait_time: Seconds to wait for page load
            max_retries: Number of retry attempts
            
        Returns:
            Dictionary with scraped data
        """
        for attempt in range(max_retries):
            try:
                self._init_driver()
                
                logger.info(f"Scraping URL (attempt {attempt + 1}/{max_retries}): {url}")
                self.driver.get(url)
                
                # Wait for page to load
                logger.info(f"Waiting {wait_time} seconds for page to load...")
                time.sleep(wait_time)
                
                # Additional wait for dynamic content
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except Exception as e:
                    logger.warning(f"Timeout waiting for body element: {e}")
                
                # Get page source
                page_source = self.driver.page_source
                soup = BeautifulSoup(page_source, 'lxml')
                
                # Remove unwanted elements
                for tag in soup(["script", "style", "nav", "footer", "header", "meta", "noscript", "iframe"]):
                    tag.decompose()
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else url
                
                # Extract main content
                main_content = (
                    soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', class_='content') or 
                    soup.find('div', id='content') or
                    soup.body
                )
                
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
                
                # Clean up text
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                cleaned_text = '\n'.join(lines)
                
                # Validate content
                if len(cleaned_text) < 100:
                    logger.warning(f"Content seems too short ({len(cleaned_text)} chars). Retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                
                logger.info(f"Successfully scraped {len(cleaned_text)} characters from {url}")
                
                return {
                    'url': url,
                    'title': title,
                    'content': cleaned_text,
                    'success': True,
                    'content_length': len(cleaned_text),
                    'extracted_at': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                    time.sleep(5)
                else:
                    return {
                        'url': url,
                        'title': '',
                        'content': '',
                        'success': False,
                        'error': str(e)
                    }
        
        return {
            'url': url,
            'title': '',
            'content': '',
            'success': False,
            'error': 'Max retries exceeded'
        }
    
    def extract_links(self, url: str, link_pattern: Optional[str] = None, wait_time: int = 10) -> List[str]:
        """
        Extract all links from a webpage
        
        Args:
            url: URL to extract links from
            link_pattern: Optional pattern to filter links (e.g., "credit-cards")
            wait_time: Seconds to wait for page load
            
        Returns:
            List of discovered URLs
        """
        try:
            self._init_driver()
            
            logger.info(f"Extracting links from: {url}")
            self.driver.get(url)
            time.sleep(wait_time)
            
            # Wait for page to load
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception as e:
                logger.warning(f"Timeout waiting for body element: {e}")
            
            # Get page source and parse
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'lxml')
            
            # Extract all links
            discovered_links = set()
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                elif not href.startswith('http'):
                    continue
                
                # Filter by pattern if provided
                if link_pattern:
                    if link_pattern in href:
                        discovered_links.add(href)
                else:
                    discovered_links.add(href)
            
            links = sorted(list(discovered_links))
            logger.info(f"Found {len(links)} links" + (f" matching pattern '{link_pattern}'" if link_pattern else ""))
            
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
    
    def close(self):
        """Close the browser"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
            finally:
                self.driver = None


class SeleniumExtractor:
    """Main extractor class that combines scraping with caching"""
    
    def __init__(self, cache_file: str = CACHE_FILE, headless: bool = True):
        self.cache_db = WebpageCacheDB(cache_file)
        self.scraper = SeleniumWebScraper(headless=headless)
    
    def extract_and_cache(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Extract webpage content and save to cache
        
        Args:
            url: URL to extract
            force_refresh: Force re-extraction even if cached
            
        Returns:
            Dictionary with extraction result
        """
        # Check cache first
        if not force_refresh and self.cache_db.has_entry(url):
            logger.info(f"URL already in cache: {url}")
            cached_entry = self.cache_db.get_entry(url)
            return {
                'success': True,
                'from_cache': True,
                'entry': cached_entry
            }
        
        # Scrape the URL
        logger.info(f"Extracting content from: {url}")
        result = self.scraper.scrape_url(url)
        
        if result['success']:
            # Add to cache
            self.cache_db.add_entry(
                url=url,
                title=result['title'],
                content=result['content'],
                metadata={
                    'content_length': result.get('content_length', 0),
                    'extracted_at': result.get('extracted_at', '')
                }
            )
            
            return {
                'success': True,
                'from_cache': False,
                'entry': self.cache_db.get_entry(url)
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error')
            }
    
    def extract_multiple(self, urls: List[str], force_refresh: bool = False):
        """Extract multiple URLs"""
        results = {}
        
        for i, url in enumerate(urls, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {i}/{len(urls)}: {url}")
            logger.info(f"{'='*60}")
            
            result = self.extract_and_cache(url, force_refresh)
            results[url] = result
            
            # Small delay between requests
            if i < len(urls):
                time.sleep(2)
        
        return results
    
    def crawl_and_extract(
        self, 
        start_url: str, 
        link_pattern: Optional[str] = None,
        max_pages: Optional[int] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Crawl a webpage and extract content from it and all discovered links
        
        Args:
            start_url: Starting URL to crawl
            link_pattern: Pattern to filter discovered links (e.g., "credit-cards")
            max_pages: Maximum number of pages to extract (None = unlimited)
            force_refresh: Force re-extraction even if cached
            
        Returns:
            Dictionary with crawling results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING CRAWL FROM: {start_url}")
        logger.info(f"Link Pattern: {link_pattern or 'None (all links)'}")
        logger.info(f"Max Pages: {max_pages or 'Unlimited'}")
        logger.info(f"{'='*70}\n")
        
        all_results = {}
        
        # Step 1: Extract the main page
        logger.info("📄 STEP 1: Extracting main page...")
        main_result = self.extract_and_cache(start_url, force_refresh)
        all_results[start_url] = main_result
        
        if not main_result['success']:
            logger.error(f"❌ Failed to extract main page: {main_result.get('error', 'Unknown error')}")
            return {
                'start_url': start_url,
                'success': False,
                'main_page_extracted': False,
                'total_pages': 0,
                'results': all_results
            }
        
        logger.info("✅ Main page extracted successfully")
        
        # Step 2: Discover links from the main page
        logger.info("\n🔍 STEP 2: Discovering links from main page...")
        discovered_links = self.scraper.extract_links(start_url, link_pattern)
        
        if not discovered_links:
            logger.warning("⚠️ No links found matching the pattern")
            return {
                'start_url': start_url,
                'success': True,
                'main_page_extracted': True,
                'links_discovered': 0,
                'total_pages': 1,
                'results': all_results
            }
        
        logger.info(f"✅ Discovered {len(discovered_links)} links")
        
        # Apply max_pages limit if specified
        if max_pages:
            # -1 because we already extracted the main page
            links_to_extract = discovered_links[:max_pages - 1]
            if len(discovered_links) > len(links_to_extract):
                logger.info(f"📊 Limited to {len(links_to_extract)} links (max_pages={max_pages})")
        else:
            links_to_extract = discovered_links
        
        # Step 3: Extract all discovered pages
        logger.info(f"\n📚 STEP 3: Extracting {len(links_to_extract)} discovered pages...\n")
        
        for i, link in enumerate(links_to_extract, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Extracting {i}/{len(links_to_extract)}: {link}")
            logger.info(f"{'='*60}")
            
            result = self.extract_and_cache(link, force_refresh)
            all_results[link] = result
            
            # Small delay between requests
            if i < len(links_to_extract):
                time.sleep(2)
        
        # Summary
        successful = sum(1 for r in all_results.values() if r['success'])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CRAWL COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total Pages Processed: {len(all_results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(all_results) - successful}")
        logger.info(f"{'='*70}\n")
        
        return {
            'start_url': start_url,
            'success': True,
            'main_page_extracted': True,
            'links_discovered': len(discovered_links),
            'links_extracted': len(links_to_extract),
            'total_pages': len(all_results),
            'successful_extractions': successful,
            'failed_extractions': len(all_results) - successful,
            'results': all_results,
            'all_discovered_links': discovered_links
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache_db.get_stats()
    
    def close(self):
        """Cleanup resources"""
        self.scraper.close()


def main():
    """Main function with example usage"""
    
    print("="*70)
    print("Selenium-Based Webpage Extractor with JSON Cache & Crawler")
    print("="*70)
    
    # Initialize extractor
    extractor = SeleniumExtractor(
        cache_file=CACHE_FILE,
        headless=False  # Set to True for headless mode
    )
    
    try:
        # Choose extraction mode
        print("\n" + "="*70)
        print("EXTRACTION MODE")
        print("="*70)
        print("1. Single/Multiple URL extraction (manual list)")
        print("2. 🕷️ CRAWL MODE: Extract main page + all linked pages")
        print("="*70)
        
        mode = input("\nSelect mode (1 or 2) [default: 2]: ").strip() or "2"
        
        if mode == "2":
            # CRAWL MODE - Recommended for credit cards pages
            print("\n" + "="*70)
            print("🕷️ CRAWL MODE ACTIVATED")
            print("="*70)
            
            # Configuration
            start_url = input("\nEnter main page URL [default: https://www.emiratesnbd.com/en/cards/credit-cards]: ").strip()
            if not start_url:
                start_url = "https://www.emiratesnbd.com/en/cards/credit-cards"
            
            link_pattern = input("Enter link pattern to filter [default: /credit-cards/]: ").strip()
            if not link_pattern:
                link_pattern = "/credit-cards/"
            
            max_pages_input = input("Max pages to extract [default: 20]: ").strip()
            max_pages = int(max_pages_input) if max_pages_input else 20
            
            force_refresh = input("Force refresh cached pages? (y/n) [default: n]: ").strip().lower() == 'y'
            
            # Execute crawl
            print("\n🚀 Starting crawl...\n")
            crawl_result = extractor.crawl_and_extract(
                start_url=start_url,
                link_pattern=link_pattern,
                max_pages=max_pages,
                force_refresh=force_refresh
            )
            
            # Display crawl results
            print("\n" + "="*70)
            print("🕷️ CRAWL RESULTS SUMMARY")
            print("="*70)
            print(f"Start URL: {crawl_result['start_url']}")
            print(f"Success: {crawl_result['success']}")
            print(f"Main Page Extracted: {crawl_result['main_page_extracted']}")
            print(f"Links Discovered: {crawl_result.get('links_discovered', 0)}")
            print(f"Links Extracted: {crawl_result.get('links_extracted', 0)}")
            print(f"Total Pages: {crawl_result.get('total_pages', 0)}")
            print(f"Successful: {crawl_result.get('successful_extractions', 0)}")
            print(f"Failed: {crawl_result.get('failed_extractions', 0)}")
            
            # Show all discovered links
            if 'all_discovered_links' in crawl_result:
                print(f"\n📋 All Discovered Links ({len(crawl_result['all_discovered_links'])}):\n")
                for i, link in enumerate(crawl_result['all_discovered_links'], 1):
                    print(f"  {i}. {link}")
        
        else:
            # MANUAL MODE - Extract specific URLs
            print("\n" + "="*70)
            print("📝 MANUAL EXTRACTION MODE")
            print("="*70)
            
            urls_to_extract = [
                "https://www.emiratesnbd.com/en/cards/credit-cards",
                # Add more URLs here as needed
            ]
            
            print(f"\nExtracting {len(urls_to_extract)} URL(s)...\n")
            results = extractor.extract_multiple(urls_to_extract, force_refresh=False)
            
            # Display results
            print("\n" + "="*70)
            print("EXTRACTION RESULTS")
            print("="*70)
            
            for url, result in results.items():
                print(f"\nURL: {url}")
                print(f"Success: {result['success']}")
                if result['success']:
                    print(f"From Cache: {result.get('from_cache', False)}")
                    entry = result['entry']
                    print(f"Title: {entry['title']}")
                    print(f"Content Length: {entry['content_length']} characters")
                    print(f"Timestamp: {entry['timestamp']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown')}")
        
        # Display final cache statistics
        print("\n" + "="*70)
        print("📊 FINAL CACHE STATISTICS")
        print("="*70)
        stats = extractor.get_cache_stats()
        print(f"Total Entries: {stats['total_entries']}")
        print(f"Total Content: {stats['total_content_chars']:,} characters")
        print(f"Cache File: {stats['cache_file']}")
        print(f"\n📄 All Cached URLs:")
        for i, url in enumerate(stats['urls'], 1):
            print(f"  {i}. {url}")
        
        print("\n" + "="*70)
        print(f"✅ Extraction complete! Cache saved to: {CACHE_FILE}")
        print("You can now upload this JSON file to GitHub for Streamlit deployment")
        print("Your RAG chatbot will have access to ALL cached pages!")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
