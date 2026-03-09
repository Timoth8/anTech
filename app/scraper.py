"""
News Article Scraper
Extracts article text from news URLs using multiple methods for reliability
"""

import trafilatura
import requests
from newspaper import Article
from bs4 import BeautifulSoup
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common Indonesian news domains (can be expanded)
INDONESIAN_NEWS_DOMAINS = [
    'kompas.com', 'detik.com', 'tempo.co', 'cnn.co.id', 
    'liputan6.com', 'tribunnews.com', 'republika.co.id',
    'antaranews.com', 'okezone.com', 'suara.com', 'cnnindonesia.com'
]

class NewsScraperException(Exception):
    """Custom exception for scraping errors"""
    pass

def extract_with_trafilatura(url: str) -> Optional[str]:
    """
    Extract article using Trafilatura (most reliable)
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            return text
    except Exception as e:
        logger.warning(f"Trafilatura extraction failed: {e}")
        return None

def extract_with_newspaper(url: str) -> Optional[str]:
    """
    Extract article using Newspaper3k (fallback method)
    """
    try:
        article = Article(url, language='id')  # Indonesian language
        article.download()
        article.parse()
        
        if article.text and len(article.text) > 100:
            return article.text
    except Exception as e:
        logger.warning(f"Newspaper3k extraction failed: {e}")
        return None

def extract_with_beautifulsoup(url: str) -> Optional[str]:
    """
    Extract article using BeautifulSoup (last resort)
    Tries to find main content in common article containers
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Try common article selectors
        selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '#article-body',
            '.content-body'
        ]
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                text = content.get_text(separator='\n', strip=True)
                if len(text) > 100:
                    return text
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        
        if len(text) > 100:
            return text
            
    except Exception as e:
        logger.warning(f"BeautifulSoup extraction failed: {e}")
        return None

def scrape_news_article(url: str) -> Dict[str, any]:
    """
    Main scraper function that tries multiple extraction methods
    
    Args:
        url: URL of the news article
        
    Returns:
        Dict containing:
            - text: Extracted article text
            - method: Extraction method used
            - url: Original URL
            - success: Boolean indicating success
            
    Raises:
        NewsScraperException: If all extraction methods fail
    """
    
    # Validate URL
    if not url or not url.startswith(('http://', 'https://')):
        raise NewsScraperException("Invalid URL format. Must start with http:// or https://")
    
    logger.info(f"Scraping article from: {url}")
    
    # Try extraction methods in order of reliability
    methods = [
        ("trafilatura", extract_with_trafilatura),
        ("newspaper3k", extract_with_newspaper),
        ("beautifulsoup", extract_with_beautifulsoup)
    ]
    
    for method_name, extract_func in methods:
        try:
            text = extract_func(url)
            if text and len(text) > 100:  # Minimum length check
                logger.info(f"Successfully extracted {len(text)} characters using {method_name}")
                return {
                    "text": text,
                    "method": method_name,
                    "url": url,
                    "success": True,
                    "length": len(text)
                }
        except Exception as e:
            logger.warning(f"{method_name} failed: {e}")
            continue
    
    # All methods failed
    raise NewsScraperException(
        "Failed to extract article text from URL. "
        "The site may be blocking scraping or the URL may not contain a news article."
    )

def validate_indonesian_content(text: str) -> bool:
    """
    Basic check if text appears to be Indonesian
    """
    # Common Indonesian words
    indonesian_indicators = [
        'yang', 'dan', 'ini', 'itu', 'untuk', 'dengan', 'dari', 
        'tidak', 'akan', 'pada', 'telah', 'adalah', 'di', 'ke'
    ]
    
    text_lower = text.lower()
    matches = sum(1 for word in indonesian_indicators if word in text_lower)
    
    # If at least 3 common Indonesian words found, likely Indonesian
    return matches >= 3

# Test function
if __name__ == "__main__":
    # Test with Indonesian news URLs
    test_urls = [
        "https://www.kompas.com/",  # Replace with actual article URL
        "https://www.detik.com/"     # Replace with actual article URL
    ]
    
    for url in test_urls:
        try:
            result = scrape_news_article(url)
            print(f"\n✅ Success!")
            print(f"Method: {result['method']}")
            print(f"Text length: {result['length']} characters")
            print(f"Preview: {result['text'][:200]}...")
            print(f"Indonesian content: {validate_indonesian_content(result['text'])}")
        except NewsScraperException as e:
            print(f"\n❌ Failed: {e}")
