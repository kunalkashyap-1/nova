"""
Web search service for retrieving context from the internet.
"""
import aiohttp
import logging
from typing import List, Dict
import os
import json
from app.utils.redis import cache_get, cache_set
import hashlib

logger = logging.getLogger(__name__)

# Cache TTLs
WEB_SEARCH_CACHE_TTL = int(os.getenv("WEB_SEARCH_CACHE_TTL", "1800"))  # 30 minutes

async def get_web_search_context(query: str, max_results: int = 5) -> List[Dict]:
    """
    Get context from web search for a given query.
    
    Args:
        query: The search query
        max_results: Maximum number of search results to retrieve
        
    Returns:
        List of context messages with web search results
    """
    # Generate a cache key for this search query
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"web_search:{query_hash}:{max_results}"
    
    # Try to get from cache first
    cached_results = await cache_get(cache_key)
    if cached_results:
        try:
            context_data = json.loads(cached_results)
            logger.info(f"Using cached web search results for: {query}")
            return context_data
        except json.JSONDecodeError:
            pass
    
    try:
        # Make a request to our search API endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/v1/tools/search",  # Using local endpoint
                json={"query": query, "max_results": max_results, "safe_search": True},
                timeout=30
            ) as response:
                if response.status != 200:
                    logger.error(f"Search API error: {response.status}")
                    return _get_fallback_context(query)
                
                data = await response.json()
                
                if not data.get("search_results"):
                    logger.warning(f"No search results found for: {query}")
                    return _get_fallback_context(query)
                
                # Format the search results into context
                search_content = "Web search results:\n\n"
                
                for i, result in enumerate(data["search_results"], 1):
                    search_content += f"[{i}] {result['title']}\n"
                    search_content += f"URL: {result['url']}\n"
                    search_content += f"{result['snippet']}\n\n"
                
                search_content += "Please use this information to provide an accurate, up-to-date answer. Include attribution to the sources, e.g. [2] or [3], when referencing specific facts."
                
                context_messages = [{
                    "role": "system",
                    "content": search_content
                }]
                
                # Cache the results
                await cache_set(cache_key, json.dumps(context_messages), ex=WEB_SEARCH_CACHE_TTL)
                
                return context_messages
                
    except Exception as e:
        logger.exception(f"Error retrieving web search results: {str(e)}")
        return _get_fallback_context(query)

def _get_fallback_context(query: str) -> List[Dict]:
    """Generate fallback context when search fails."""
    return [{
        "role": "system",
        "content": f"Unable to retrieve web search results for: {query}. Please use your knowledge to provide the best possible answer, but note that your information may not be current."
    }]
