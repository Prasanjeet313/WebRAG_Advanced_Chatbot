"""
Webpage RAG Chatbot - Enhanced with JSON Cache Support
Streamlit Cloud Compatible Version with Pre-extracted Data Loading
Uses LangChain, FAISS, and Groq LLM (Pure Python Scraping + Cache)
"""

import os
import streamlit as st
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import logging
import time
import json
import hashlib
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime

from dotenv import load_dotenv
# Load env vars
load_dotenv()

# Optional Cloudflare-bypass libraries
try:
    import curl_cffi.requests as cc_requests
    CURL_CFFI_AVAILABLE = True
except Exception:
    cc_requests = None
    CURL_CFFI_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

# --- MODERNIZED IMPORTS FOR LANGCHAIN 1.x (DO NOT CHANGE - Streamlit compatible) ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Use langchain_classic for the legacy chains
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache file configuration
CACHE_FILE = "webpage_cache.json"

@dataclass
class Config:
    """Configuration class for the application"""
    # Use st.secrets for Cloud deployment compatibility, fallback to os.environ
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "outputs")
    # scraper backend: 'auto' | 'requests' | 'curl_cffi' | 'playwright'
    SCRAPER_BACKEND: str = os.environ.get("SCRAPER_BACKEND", "auto")
    # Playwright headless mode
    PLAYWRIGHT_HEADLESS: bool = os.environ.get("PLAYWRIGHT_HEADLESS", "1") == "1"
    # Cache file path
    CACHE_FILE: str = os.environ.get("CACHE_FILE", CACHE_FILE)

# Default output directory (can be overridden via env or Config)
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def url_to_filename(url: str, suffix: str = "") -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{h}{suffix}"


def save_bytes(output_dir: str, filename: str, data: bytes):
    ensure_dir(output_dir)
    Path(output_dir, filename).write_bytes(data)


def save_text(output_dir: str, filename: str, text: str):
    ensure_dir(output_dir)
    Path(output_dir, filename).write_text(text, encoding='utf-8')


def save_json(output_dir: str, filename: str, obj):
    ensure_dir(output_dir)
    Path(output_dir, filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def save_pickle(output_dir: str, filename: str, obj):
    ensure_dir(output_dir)
    with open(Path(output_dir, filename), 'wb') as f:
        pickle.dump(obj, f)


def save_chat_history(output_dir: str, chat_history):
    try:
        filename = f"chat_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        save_json(os.path.join(output_dir, 'chats'), filename, chat_history)
    except Exception as e:
        logger.warning(f"Could not save chat history: {e}")


class WebpageCacheLoader:
    """Loads pre-extracted webpage data from JSON cache"""
    
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from JSON file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ Loaded cache with {len(data)} entries from {self.cache_file}")
                    return data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return {}
        else:
            logger.warning(f"⚠️ Cache file {self.cache_file} not found")
            return {}
    
    def has_url(self, url: str) -> bool:
        """Check if URL exists in cache"""
        return url in self.cache_data
    
    def get_cached_data(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached data for a URL"""
        if self.has_url(url):
            entry = self.cache_data[url]
            logger.info(f"✅ Found cached data for: {url}")
            logger.info(f"   Title: {entry.get('title', 'N/A')}")
            logger.info(f"   Content Length: {entry.get('content_length', 0)} chars")
            logger.info(f"   Extracted: {entry.get('timestamp', 'N/A')}")
            return {
                'url': url,
                'title': entry.get('title', ''),
                'content': entry.get('content', ''),
                'success': True,
                'from_cache': True,
                'cached_at': entry.get('timestamp', ''),
                'extracted_with': entry.get('extracted_with', 'unknown')
            }
        return None
    
    def list_cached_urls(self) -> List[str]:
        """Get list of all cached URLs"""
        return list(self.cache_data.keys())
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            'total_entries': len(self.cache_data),
            'cache_file': str(self.cache_file),
            'cache_exists': self.cache_file.exists(),
            'urls': self.list_cached_urls()
        }


class WebScraper:
    """
    Streamlit Cloud Compatible Scraper with Cache Support
    Supports requests, curl_cffi, or Playwright to bypass Cloudflare when needed.
    """
    def __init__(self, config: Optional[Config] = None, cache_loader: Optional[WebpageCacheLoader] = None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.config = config or Config()
        self.backend = (self.config.SCRAPER_BACKEND or 'auto').lower()
        self.playwright_headless = getattr(self.config, 'PLAYWRIGHT_HEADLESS', True)
        self.cache_loader = cache_loader

    def _looks_like_cloudflare(self, text: str, status: int) -> bool:
        if status in (403, 429, 503):
            return True
        if not text:
            return True
        low = text.lower()
        checks = [
            'cloudflare',
            'checking your browser',
            'attention required',
            'cf-chl-bypass',
            'hit 1 sec'
        ]
        return any(c in low for c in checks)

    def _fetch_requests(self, url: str, timeout: int = 15):
        try:
            resp = requests.get(url, headers=self.headers, timeout=timeout)
            return {'status_code': getattr(resp, 'status_code', 200), 'content': getattr(resp, 'content', b''), 'text': getattr(resp, 'text', '')}
        except Exception as e:
            raise

    def _fetch_curl_cffi(self, url: str, timeout: int = 15):
        if not CURL_CFFI_AVAILABLE:
            raise ImportError('curl_cffi not available')
        resp = cc_requests.get(url, headers=self.headers, timeout=timeout)
        return {'status_code': getattr(resp, 'status_code', 200), 'content': getattr(resp, 'content', b''), 'text': getattr(resp, 'text', '')}

    def _fetch_playwright(self, url: str, timeout: int = 30):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError('Playwright not available')
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.playwright_headless)
            page = browser.new_page(user_agent=self.headers.get('User-Agent'))
            try:
                page.set_default_navigation_timeout(timeout * 1000)
                page.goto(url, wait_until='networkidle')
                content = page.content()
                # Playwright does not expose status easily for page content; assume 200 on success
                return {'status_code': 200, 'content': content.encode('utf-8'), 'text': content}
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a given URL
        First checks cache, then falls back to live scraping
        """
        try:
            # 🔍 CHECK CACHE FIRST
            if self.cache_loader:
                cached_data = self.cache_loader.get_cached_data(url)
                if cached_data:
                    st.info(f"📦 Loading from cache (extracted with {cached_data.get('extracted_with', 'unknown')})")
                    return cached_data
            
            # If not in cache, proceed with live scraping
            logger.info(f"🌐 Cache miss - scraping URL: {url} using backend={self.backend}")
            st.warning("⚠️ URL not in cache - attempting live scraping (may fail on Cloudflare-protected sites)")

            last_exc = None
            resp = None

            # Helper to try a fetcher and set resp
            def try_fetch(fetch_fn, *a, **kw):
                nonlocal last_exc, resp
                try:
                    resp = fetch_fn(*a, **kw)
                    return True
                except Exception as e:
                    last_exc = e
                    logger.debug(f"Fetcher {fetch_fn.__name__} failed: {e}")
                    return False

            # Decide order
            tried = []
            if self.backend == 'requests':
                try_fetch(self._fetch_requests, url)
                tried = ['requests']
            elif self.backend == 'curl_cffi':
                try_fetch(self._fetch_curl_cffi, url)
                tried = ['curl_cffi']
            elif self.backend == 'playwright':
                try_fetch(self._fetch_playwright, url)
                tried = ['playwright']
            else:  # auto
                # Try requests first
                if try_fetch(self._fetch_requests, url):
                    tried.append('requests')
                    if self._looks_like_cloudflare(resp.get('text', ''), resp.get('status_code', 200)):
                        logger.info('Detected possible Cloudflare/JS protection; trying curl_cffi')
                        if CURL_CFFI_AVAILABLE and try_fetch(self._fetch_curl_cffi, url):
                            tried.append('curl_cffi')
                        elif PLAYWRIGHT_AVAILABLE and try_fetch(self._fetch_playwright, url):
                            tried.append('playwright')
                else:
                    # try curl_cffi then playwright
                    if CURL_CFFI_AVAILABLE and try_fetch(self._fetch_curl_cffi, url):
                        tried.append('curl_cffi')
                    elif PLAYWRIGHT_AVAILABLE and try_fetch(self._fetch_playwright, url):
                        tried.append('playwright')

            if not resp:
                raise last_exc or RuntimeError('All fetchers failed')

            content_bytes = resp.get('content', b'') if resp.get('content', None) is not None else (resp.get('text', '') or '').encode('utf-8')
            soup = BeautifulSoup(content_bytes, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
                script.decompose()

            # Get title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else url

            # Get main content
            # Try to find specific content containers first
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Clean up text (remove excessive newlines)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            if len(text) < 100:
                logger.warning("Scraped content is very short. Page might be JS-rendered or Cloudflare-protected.")
                st.warning("⚠️ Content appears incomplete - consider using Selenium extractor locally")

            logger.info(f"Successfully scraped {len(text)} characters from {url} (via {','.join(tried)})")

            # Save artifacts: raw HTML, cleaned text, and metadata
            try:
                html_filename = url_to_filename(url, '.html')
                save_bytes(os.path.join(DEFAULT_OUTPUT_DIR, 'html'), html_filename, content_bytes)
                text_filename = url_to_filename(url, '.txt')
                save_text(os.path.join(DEFAULT_OUTPUT_DIR, 'text'), text_filename, text)
                metadata_filename = url_to_filename(url, '.json')
                meta = {'url': url, 'title': title_text, 'chars': len(text), 'saved_at': datetime.utcnow().isoformat(), 'backend_tried': tried}
                save_json(os.path.join(DEFAULT_OUTPUT_DIR, 'metadata'), metadata_filename, meta)
            except Exception as e:
                logger.warning(f"Could not save scraped files: {e}")

            return {
                'url': url,
                'title': title_text,
                'content': text,
                'success': True,
                'from_cache': False
            }
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'success': False,
                'error': str(e)
            }

class DocumentProcessor:
    """Handles document chunking and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_scraped_data(self, scraped_data: Dict[str, Any]) -> List[Document]:
        if not scraped_data.get('success'):
            return []
        
        metadata = {
            'source': scraped_data['url'],
            'title': scraped_data['title'],
            'from_cache': scraped_data.get('from_cache', False)
        }
        
        document = Document(
            page_content=scraped_data['content'],
            metadata=metadata
        )
        
        chunks = self.text_splitter.split_documents([document])

        # Save chunks to disk as JSON
        try:
            chunks_list = []
            for i, c in enumerate(chunks):
                chunks_list.append({
                    'index': i,
                    'content': c.page_content,
                    'metadata': c.metadata
                })
            chunks_filename = url_to_filename(scraped_data['url'], '_chunks.json')
            save_json(os.path.join(DEFAULT_OUTPUT_DIR, 'chunks'), chunks_filename, chunks_list)
        except Exception as e:
            logger.warning(f"Could not save chunks: {e}")

        return chunks

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self, embedding_model: str, use_gpu: bool = False):
        # Force CPU for Streamlit Cloud to avoid memory/compatibility issues
        device = 'cpu'
        logger.info(f"Initializing embeddings on {device}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store: Optional[FAISS] = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("No documents provided")
        
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        # Save vector store and embeddings
        try:
            store_name = url_to_filename('vectorstore', '')
            store_dir = Path(DEFAULT_OUTPUT_DIR) / 'vector_store' / store_name
            ensure_dir(store_dir.parent)
            # Try to use FAISS save_local if available
            if hasattr(self.vector_store, "save_local"):
                self.vector_store.save_local(str(store_dir))
                logger.info(f"Saved vector store to {store_dir}")
            else:
                # Fallback to pickle
                save_pickle(str(Path(DEFAULT_OUTPUT_DIR) / 'vector_store'), f"{store_name}.pkl", self.vector_store)
                logger.info(f"Pickled vector store to {Path(DEFAULT_OUTPUT_DIR) / 'vector_store' / (store_name + '.pkl')}")
        except Exception as e:
            logger.warning(f"Could not save vector store: {e}")

        # Save embeddings separately
        try:
            texts = [d.page_content for d in documents]
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_arr = np.array(embeddings, dtype=np.float32)
            embed_filename = url_to_filename('embeddings', '.npy')
            ensure_dir(Path(DEFAULT_OUTPUT_DIR) / 'embeddings')
            np.save(Path(DEFAULT_OUTPUT_DIR) / 'embeddings' / embed_filename, embeddings_arr)
            # Save metadata mapping
            meta = [{'index': i, 'metadata': d.metadata} for i, d in enumerate(documents)]
            save_json(os.path.join(DEFAULT_OUTPUT_DIR, 'embeddings'), url_to_filename('embeddings_meta', '.json'), meta)
            logger.info(f"Saved embeddings to {Path(DEFAULT_OUTPUT_DIR) / 'embeddings' / embed_filename}")
        except Exception as e:
            logger.warning(f"Could not save embeddings: {e}")

        return self.vector_store
    
    def get_retriever(self, k: int = 10):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

class RAGChatbot:
    """Main RAG chatbot using LangChain and Groq"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Check API Key
        if not config.GROQ_API_KEY:
            st.error("Groq API Key not found! Please add it to secrets or .env")
            st.stop()

        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        
    def create_qa_chain(self, retriever):
        template = """You are a helpful AI assistant specialized in answering questions about webpage content.
Use the following pieces of context from the webpage to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always provide detailed and accurate answers based on the webpage content.

Context chunks from the webpage:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return self.qa_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {'success': False, 'answer': "Please load a webpage first."}
            
        try:
            result = self.qa_chain({"question": question})
            return {
                'answer': result['answer'],
                'source_documents': result.get('source_documents', []),
                'success': True
            }
        except Exception as e:
            return {'success': False, 'answer': f"Error: {str(e)}"}
    
    def clear_memory(self):
        self.memory.clear()

class WebpageRAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        # Initialize cache loader
        self.cache_loader = WebpageCacheLoader(config.CACHE_FILE)
        # Pass cache loader to scraper
        self.scraper = WebScraper(config, cache_loader=self.cache_loader)
        self.processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_manager = VectorStoreManager(config.EMBEDDING_MODEL)
        self.chatbot = RAGChatbot(config)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return self.cache_loader.get_cache_info()
    
    def process_url(self, url: str) -> Dict[str, Any]:
        try:
            scraped_data = self.scraper.scrape_url(url)
            if not scraped_data['success']:
                return {'success': False, 'error': scraped_data.get('error')}
            
            documents = self.processor.process_scraped_data(scraped_data)
            if not documents:
                return {'success': False, 'error': 'No content found on page'}
            
            self.vector_manager.create_vector_store(documents)
            retriever = self.vector_manager.get_retriever()
            self.chatbot.create_qa_chain(retriever)
            
            return {
                'success': True,
                'title': scraped_data['title'],
                'num_chunks': len(documents),
                'content_length': len(scraped_data['content']),
                'from_cache': scraped_data.get('from_cache', False)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def process_multiple_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process multiple URLs and combine them into a single vector store
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            all_documents = []
            processed_urls = []
            failed_urls = []
            url_info = []
            
            logger.info(f"Processing {len(urls)} URLs...")
            
            for i, url in enumerate(urls, 1):
                logger.info(f"Processing {i}/{len(urls)}: {url}")
                
                try:
                    scraped_data = self.scraper.scrape_url(url)
                    
                    if scraped_data['success']:
                        documents = self.processor.process_scraped_data(scraped_data)
                        
                        if documents:
                            all_documents.extend(documents)
                            processed_urls.append(url)
                            url_info.append({
                                'url': url,
                                'title': scraped_data['title'],
                                'chunks': len(documents),
                                'content_length': len(scraped_data['content']),
                                'from_cache': scraped_data.get('from_cache', False)
                            })
                            logger.info(f"✅ Processed: {url} ({len(documents)} chunks)")
                        else:
                            failed_urls.append({'url': url, 'error': 'No content found'})
                            logger.warning(f"⚠️ No content: {url}")
                    else:
                        failed_urls.append({'url': url, 'error': scraped_data.get('error', 'Unknown error')})
                        logger.warning(f"❌ Failed: {url}")
                        
                except Exception as e:
                    failed_urls.append({'url': url, 'error': str(e)})
                    logger.error(f"❌ Error processing {url}: {e}")
            
            if not all_documents:
                return {
                    'success': False,
                    'error': 'No documents could be processed',
                    'processed_urls': [],
                    'failed_urls': failed_urls
                }
            
            # Create vector store with all documents
            logger.info(f"Creating vector store with {len(all_documents)} total chunks from {len(processed_urls)} pages...")
            self.vector_manager.create_vector_store(all_documents)
            retriever = self.vector_manager.get_retriever()
            self.chatbot.create_qa_chain(retriever)
            
            return {
                'success': True,
                'total_urls': len(urls),
                'processed_urls': processed_urls,
                'failed_urls': failed_urls,
                'total_chunks': len(all_documents),
                'url_info': url_info
            }
            
        except Exception as e:
            logger.error(f"Error processing multiple URLs: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_all_cached_pages(self) -> Dict[str, Any]:
        """
        Load all pages from cache into the vector store
        
        Returns:
            Dictionary with loading results
        """
        cached_urls = self.cache_loader.list_cached_urls()
        
        if not cached_urls:
            return {
                'success': False,
                'error': 'No cached pages found',
                'processed_urls': []
            }
        
        logger.info(f"Loading all {len(cached_urls)} cached pages...")
        return self.process_multiple_urls(cached_urls)

    def ask(self, question: str):
        return self.chatbot.ask(question)
    
    def clear_memory(self):
        self.chatbot.clear_memory()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Webpage RAG Chatbot (Cache-Enhanced)",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Webpage RAG Chatbot")
    st.markdown("### Enhanced with Pre-extracted Data Cache")
    
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'webpage_loaded' not in st.session_state:
        st.session_state.webpage_loaded = False
    if 'loaded_pages' not in st.session_state:
        st.session_state.loaded_pages = []
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display cache information
        if st.session_state.pipeline is None:
            # Create a temporary pipeline just to check cache
            temp_pipeline = WebpageRAGPipeline(st.session_state.config)
            cache_info = temp_pipeline.get_cache_info()
        else:
            cache_info = st.session_state.pipeline.get_cache_info()
        
        # Cache Status
        st.subheader("📦 Cache Status")
        if cache_info['cache_exists']:
            st.success(f"✅ Cache loaded: {cache_info['total_entries']} entries")
            if cache_info['urls']:
                with st.expander("View cached URLs"):
                    for url in cache_info['urls']:
                        st.caption(f"• {url}")
        else:
            st.warning("⚠️ No cache file found")
            st.caption(f"Expected at: {cache_info['cache_file']}")
            st.info("💡 Use selenium_extractor.py to create cache")
        
        st.divider()
        
        # URL Input
        st.subheader("🔗 Load Pages")
        
        # Option 1: Load single URL
        url = st.text_input(
            "Single URL",
            value="https://www.emiratesnbd.com/en/cards/credit-cards",
            help="Enter URL (will use cache if available)"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Load Single", type="secondary", use_container_width=True):
                with st.spinner("Processing..."):
                    pipeline = WebpageRAGPipeline(st.session_state.config)
                    result = pipeline.process_url(url)
                    
                    if result['success']:
                        st.session_state.pipeline = pipeline
                        st.session_state.webpage_loaded = True
                        st.session_state.chat_history = []
                        st.session_state.loaded_pages = [{
                            'url': url,
                            'title': result['title'],
                            'chunks': result['num_chunks'],
                            'from_cache': result.get('from_cache', False)
                        }]
                        
                        # Show result with cache indicator
                        if result.get('from_cache'):
                            st.success(f"✅ Loaded from cache!")
                        else:
                            st.success(f"✅ Scraped successfully!")
                        
                        st.info(f"📄 **Title:** {result['title']}")
                        st.info(f"📊 **Chunks:** {result['num_chunks']}")
                    else:
                        st.error(f"❌ Error: {result.get('error')}")
                        st.session_state.webpage_loaded = False
        
        with col2:
            # Option 2: Load ALL cached pages
            if st.button("📦 Load ALL Cached", type="primary", use_container_width=True):
                with st.spinner("Loading all cached pages..."):
                    pipeline = WebpageRAGPipeline(st.session_state.config)
                    result = pipeline.load_all_cached_pages()
                    
                    if result['success']:
                        st.session_state.pipeline = pipeline
                        st.session_state.webpage_loaded = True
                        st.session_state.chat_history = []
                        st.session_state.loaded_pages = result.get('url_info', [])
                        
                        st.success(f"✅ Loaded {len(result['processed_urls'])} pages!")
                        st.info(f"📊 **Total Chunks:** {result['total_chunks']}")
                        
                        if result.get('failed_urls'):
                            st.warning(f"⚠️ {len(result['failed_urls'])} pages failed to load")
                    else:
                        st.error(f"❌ Error: {result.get('error')}")
                        st.session_state.webpage_loaded = False
        
        # Display loaded pages
        if st.session_state.webpage_loaded and st.session_state.loaded_pages:
            st.divider()
            st.subheader("📚 Loaded Pages")
            st.caption(f"**{len(st.session_state.loaded_pages)} page(s) in vector store**")
            
            with st.expander(f"View loaded pages ({len(st.session_state.loaded_pages)})", expanded=False):
                for i, page in enumerate(st.session_state.loaded_pages, 1):
                    cache_badge = "📦" if page.get('from_cache') else "🌐"
                    st.caption(f"{i}. {cache_badge} **{page.get('title', 'Untitled')}**")
                    st.caption(f"   └─ {page.get('chunks', 0)} chunks | {page.get('url', '')[:50]}...")

        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            if st.session_state.pipeline:
                st.session_state.pipeline.clear_memory()
            st.rerun()
        
        st.divider()
        st.caption(f"**LLM:** {st.session_state.config.GROQ_MODEL}")
        st.caption(f"**Embeddings:** {st.session_state.config.EMBEDDING_MODEL}")

    # Chat Interface
    if st.session_state.webpage_loaded:
        st.markdown("---")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg and msg["role"] == "assistant":
                    with st.expander("📚 View Sources (Click to see which pages were used)"):
                        for i, doc in enumerate(msg["sources"][:5], 1):
                            st.caption(f"**Source {i}:** 🔗 {doc.metadata.get('source', 'Unknown')}")
                            st.caption(f"**Title:** {doc.metadata.get('title', 'N/A')}")
                            st.text(doc.page_content[:300] + "...")

        if prompt := st.chat_input("Ask about the webpage..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.pipeline.ask(prompt)
                    st.write(response['answer'])
                    
                    if response.get('source_documents'):
                        with st.expander("📚 View Sources (Click to see which pages were used)"):
                            for i, doc in enumerate(response['source_documents'][:5], 1):
                                st.caption(f"**Source {i}:** 🔗 {doc.metadata.get('source', 'Unknown')}")
                                st.caption(f"**Title:** {doc.metadata.get('title', 'N/A')}")
                                st.text(doc.page_content[:300] + "...")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "sources": response.get('source_documents')
                    })

                    # Save chat history
                    try:
                        save_chat_history(st.session_state.config.OUTPUT_DIR, st.session_state.chat_history)
                    except Exception as e:
                        logger.warning(f"Could not save chat history: {e}")
    else:
        st.info("👈 Please load webpage(s) from the sidebar to start chatting!")
        
        st.markdown("""
        ### 🚀 How to use:
        
        **Option 1: Load ALL cached pages (Recommended for comprehensive answers)**
        1. Click "📦 Load ALL Cached" to load all pages at once
        2. The chatbot will have access to ALL credit card details
        3. Ask questions about any specific card or compare cards
        
        **Option 2: Load single page**
        1. Enter a specific URL
        2. Click "🔄 Load Single" to load just that page
        3. Best for focused queries on one page
        
        ### 💡 Benefits of Loading All Pages:
        - ✅ Answer questions about ANY credit card from their detail pages
        - ✅ Compare multiple credit cards in one query
        - ✅ Get comprehensive answers from all available data
        - ✅ Source attribution shows which specific page the answer came from
        
        ### 📝 Example Questions (after loading all pages):
        - What are the benefits of the Visa Infinite credit card?
        - Compare Voyager World and Voyager World Elite cards
        - Which cards offer airport lounge access?
        - What's the annual fee for the SHARE Visa Signature card?
        - Tell me about the Marriott Bonvoy cards
        """)


if __name__ == "__main__":
    main()
