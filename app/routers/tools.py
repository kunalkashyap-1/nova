from fastapi import APIRouter, Depends, HTTPException
from app.schemas.tool import ToolInput, ToolOutput, SearchResult
from app.routers.memories import get_chroma_client
import aiohttp
import asyncio
import logging
import os
import json
from typing import List, Optional
from bs4 import BeautifulSoup
import bs4 as _bs4  # alias for fallback search
import ollama
import re
import hashlib
from app.utils.redis import cache_get, cache_set
from datetime import timedelta

LLM_MODEL = os.getenv("BACKEND_LLM_MODEL", "deepseek-r1:1.5b")

# Configure logging
logger = logging.getLogger(__name__)

# Load API keys from environment
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

router = APIRouter(prefix="/api/v1/tools", tags=["Tools"])

@router.post("/memory_lookup", response_model=ToolOutput)
async def memory_lookup(tool_input: ToolInput, chroma_client=Depends(get_chroma_client), user_id: str = ""):
    # get all memories from chroma
    # return the memories as a list of MemoryOut objects (ToolOutput)
    return ToolOutput(result="Memory lookup not implemented yet")


async def fetch_url_content(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch content from a URL with error handling and timeouts."""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
            else:
                logger.warning(f"Failed to fetch {url}: Status {response.status}")
                return ""
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return ""

async def parse_webpage(html: str, query: str) -> str:
    """Extract and summarize relevant content from a webpage."""
    if not html:
        return ""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "aside"]):
        script.extract()
    
    # Get text content
    text = soup.get_text(separator=" ", strip=True)
    
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Truncate to a reasonable size (first 2000 characters)
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    return text

async def _search_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    """Lightweight fallback search scraping DuckDuckGo HTML results (no API key required)."""
    results: List[SearchResult] = []
    try:
        async with aiohttp.ClientSession() as session:
            params = {"q": query, "kl": "us-en"}
            async with session.get("https://duckduckgo.com/html/", params=params, timeout=15) as resp:
                if resp.status != 200:
                    logger.warning("DuckDuckGo request failed: %s", resp.status)
                    return []
                html = await resp.text()
                soup = _bs4.BeautifulSoup(html, "html.parser")
                for a in soup.select("a.result__a")[:max_results]:
                    title = a.get_text()
                    url = a.get("href")
                    snippet_tag = a.find_parent("div", class_="result")
                    snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
                    results.append(SearchResult(title=title, snippet=snippet[:256], url=url, source="duckduckgo"))
        return results
    except Exception as exc:
        logger.error("DuckDuckGo scraping error: %s", exc)
        return []

async def search_serp_api(query: str, max_results: int = 5, safe_search: bool = True) -> List[SearchResult]:
    """Search the web using SerpAPI."""
    # If no SERPAPI_KEY configured, immediately fall back to DuckDuckGo scraping
    if not SERPAPI_KEY:
        return await _search_duckduckgo(query, max_results)
    
    # Create a cache key based on the query and parameters
    query_string = f"{query}:{max_results}:{safe_search}"
    cache_key = f"search:{hashlib.md5(query_string.encode()).hexdigest()}"
    
    # Try to get results from cache first
    cached_results = await cache_get(cache_key)
    if cached_results:
        return json.loads(cached_results)
    
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": max_results,
            "safe": "active" if safe_search else "off"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get("https://serpapi.com/search", params=params) as response:
                if response.status != 200:
                    logger.error(f"SerpAPI error: {response.status}")
                    return []
                
                data = await response.json()
                
                if "error" in data:
                    logger.error(f"SerpAPI returned error: {data['error']}")
                    return []
                
                results = []
                if "organic_results" in data:
                    for result in data["organic_results"][:max_results]:
                        results.append(SearchResult(
                            title=result.get("title", ""),
                            snippet=result.get("snippet", ""),
                            url=result.get("link", ""),
                            source="serpapi"
                        ))
                
                # Cache the results for 1 hour
                await cache_set(cache_key, json.dumps([r.dict() for r in results]), timedelta(hours=1))
                
                return results
    except Exception as e:
        logger.error(f"Error searching with SerpAPI: {str(e)}")
        return []

async def search_and_extract(query: str, max_results: int = 5, safe_search: bool = True) -> List[SearchResult]:
    """Search the web (SerpAPI → fallback DuckDuckGo) and scrape content."""
    """Search the web and extract content from top results."""
    search_results = await search_serp_api(query, max_results, safe_search)
    
    if not search_results:
        # Final attempt: DuckDuckGo fallback if not already used
        search_results = await _search_duckduckgo(query, max_results)
    if not search_results:
        return []
    
    async with aiohttp.ClientSession() as session:
        content_tasks = []
        for result in search_results:
            task = asyncio.create_task(fetch_url_content(session, result.url))
            content_tasks.append((result, task))
        
        # Wait for all fetches to complete
        for result, task in content_tasks:
            try:
                html = await task
                extract = await parse_webpage(html, query)
                # Enhance the snippet with extracted content if we have it
                if extract:
                    result.snippet = f"{result.snippet} {extract[:500]}"
            except Exception as e:
                logger.error(f"Error processing {result.url}: {str(e)}")
    
    return search_results

@router.post("/llm_query", response_model=ToolOutput)
async def llm_query(tool_input: ToolInput):
    """Generic backend LLM call using Ollama and `deepseek-r1:1.5b` (or env override)"""
    if not tool_input.query:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": tool_input.query.strip()}],
            temperature=0.7,
            top_p=0.95,
        )
        content: Optional[str] = resp.get("message", {}).get("content") if isinstance(resp, dict) else None
        return ToolOutput(result=content or "")
    except Exception as exc:
        logger.error("LLM query failed: %s", exc)
        raise HTTPException(status_code=500, detail="LLM query failed")


@router.post("/search", response_model=ToolOutput)
async def search(tool_input: ToolInput):
    """Search the web for information related to the query."""
    if not tool_input.query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        search_results = await search_and_extract(
            query=tool_input.query,
            max_results=tool_input.max_results,
            safe_search=tool_input.safe_search
        )
        
        return ToolOutput(
            result=f"Found {len(search_results)} results for '{tool_input.query}'.",
            search_results=search_results
        )
    except Exception as e:
        logger.exception(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")