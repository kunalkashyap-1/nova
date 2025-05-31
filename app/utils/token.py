"""
Utility functions for token estimation and context optimization.
"""
import hashlib
import os
import tiktoken
import re
from functools import lru_cache
from typing import List, Dict, Any, Optional

# Load environment variables
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "30000"))
DEFAULT_ENCODING = os.getenv("DEFAULT_ENCODING", "cl100k_base")  # Default for most newer models
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Map model names to their encoding types
MODEL_TO_ENCODING = {
    # OpenAI models
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    # Anthropic models - map to a similar tokenizer for estimation
    "claude-2": "cl100k_base",
    "claude-instant": "cl100k_base",
    # Ollama models (using tiktoken's encodings as approximations)
    "llama2": "cl100k_base",
    "mistral": "cl100k_base",
    "codellama": "cl100k_base",
    "vicuna": "cl100k_base",
}


@lru_cache(maxsize=128)
def get_tokenizer(model_name: Optional[str] = None):
    """Get the appropriate tokenizer for a model."""
    encoding_name = DEFAULT_ENCODING
    
    # Check if we have a specific encoding for this model
    if model_name:
        # Extract base model name from potential fine-tuned version
        base_model = model_name.split(':')[-1]
        # Try to get encoding for this model
        encoding_name = MODEL_TO_ENCODING.get(base_model, DEFAULT_ENCODING)
    
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        # Fall back to DEFAULT_ENCODING if the requested one isn't available
        if encoding_name != DEFAULT_ENCODING:
            try:
                return tiktoken.get_encoding(DEFAULT_ENCODING)
            except Exception:
                # Last resort: use the simplest approximation
                return None
        return None


def estimate_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Estimate the number of tokens in a text using the appropriate tokenizer.
    
    Args:
        text: The text to estimate tokens for
        model_name: Optional model name to use specific tokenizer
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
        
    # Get the appropriate tokenizer
    encoder = get_tokenizer(model_name)
    
    if encoder:
        try:
            # Use proper tokenization
            return len(encoder.encode(text))
        except Exception:
            # Fall back to approximation if tokenization fails
            pass
    
    # Fallback estimation if tokenizer is unavailable
    # Different models have different ratios of tokens to characters
    # This is a rough approximation
    return len(text) // 4  # Rough estimate: 1 token ~= 4 chars in English


def optimize_context(
    context_docs: List[str], 
    query: str, 
    max_tokens: int = DEFAULT_MAX_TOKENS // 2,
    model_name: Optional[str] = None
) -> List[str]:
    """
    Optimize context by:
    1. Removing duplicate or near-duplicate content
    2. Prioritizing more relevant content
    3. Ensuring we stay within token limits
    4. Chunking large documents if needed
    
    Args:
        context_docs: List of document contents to optimize
        query: The query to optimize context for
        max_tokens: Maximum tokens to include in context
        model_name: Optional model name for tokenizer selection
        
    Returns:
        Optimized list of document contents
    """
    if not context_docs:
        return []

    # Simple deduplication by keeping track of content hashes
    seen_hashes = set()
    
    # Use shorter hash for faster comparison
    def get_hash(text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
    
    # Extract key terms from query for improved matching (remove common words)
    stop_words = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "by", "to", "and", "or", "is", "are"}
    query_terms = [term.lower() for term in re.findall(r'\b\w+\b', query) if term.lower() not in stop_words]
    
    docs_with_scores = []
    
    for doc in context_docs:
        doc_hash = get_hash(doc)
        
        # Skip if we've seen very similar content
        if doc_hash in seen_hashes:
            continue
            
        seen_hashes.add(doc_hash)
        
        # Multiple relevance signals
        # 1. Term frequency - how many query terms appear in the document
        doc_lower = doc.lower()
        term_matches = sum(1 for term in query_terms if term in doc_lower)
        term_score = term_matches / len(query_terms) if query_terms else 0
        
        # 2. Term density - how concentrated the matches are
        doc_word_count = len(re.findall(r'\b\w+\b', doc_lower))
        density_score = term_matches / doc_word_count if doc_word_count > 0 else 0
        
        # 3. Exact phrase matching - bonus for exact phrase matches
        phrase_score = 0.3 if query.lower() in doc_lower else 0
        
        # Combined relevance score (weighted combination)
        combined_score = (term_score * 0.5) + (density_score * 0.3) + phrase_score
        
        # Estimate tokens
        token_estimate = estimate_tokens(doc, model_name)
        
        # If document is too large, consider splitting it
        if token_estimate > max_tokens // 3:  # If single doc is > 1/3 of our budget
            # Split the document and score each chunk
            chunks = chunk_text(doc, model_name)
            
            for i, chunk in enumerate(chunks):
                # Re-score the chunk
                chunk_lower = chunk.lower()
                chunk_matches = sum(1 for term in query_terms if term in chunk_lower)
                chunk_words = len(re.findall(r'\b\w+\b', chunk_lower))
                
                chunk_term_score = chunk_matches / len(query_terms) if query_terms else 0
                chunk_density = chunk_matches / chunk_words if chunk_words > 0 else 0
                chunk_phrase = 0.3 if query.lower() in chunk_lower else 0
                
                chunk_score = (chunk_term_score * 0.5) + (chunk_density * 0.3) + chunk_phrase
                chunk_tokens = estimate_tokens(chunk, model_name)
                
                docs_with_scores.append((chunk, chunk_score, chunk_tokens))
        else:
            docs_with_scores.append((doc, combined_score, token_estimate))
        
    # Sort by relevance score (descending)
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take documents until we hit token limit, but ensure diversity
    optimized_docs = []
    current_tokens = 0
    min_docs_to_include = min(3, len(docs_with_scores))  # Always include at least top 3 if available
    
    # First, include minimum number of most relevant docs regardless of token limits
    for i in range(min_docs_to_include):
        if i < len(docs_with_scores):
            doc, _, tokens = docs_with_scores[i]
            # Truncate if a single doc is too large
            if tokens > max_tokens // 2:
                # Truncate to half our budget
                doc = truncate_to_token_limit(doc, max_tokens // 2, model_name)
                tokens = estimate_tokens(doc, model_name)
            
            optimized_docs.append(doc)
            current_tokens += tokens
            
    # Then add more until we hit the token limit
    for doc, _, tokens in docs_with_scores[min_docs_to_include:]:
        if current_tokens + tokens <= max_tokens:
            optimized_docs.append(doc)
            current_tokens += tokens
        else:
            # Try to fit a truncated version if it's valuable
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 100:  # Only if we have a meaningful amount left
                truncated = truncate_to_token_limit(doc, remaining_tokens, model_name)
                optimized_docs.append(truncated)
            break
            
    return optimized_docs


def chunk_text(text: str, model_name: Optional[str] = None) -> List[str]:
    """
    Split text into semantically meaningful chunks.
    
    Args:
        text: The text to chunk
        model_name: Optional model name for tokenizer
        
    Returns:
        List of text chunks
    """
    # Simple paragraph-based chunking
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
            
        para_tokens = estimate_tokens(para_stripped, model_name)
        
        # If paragraph itself exceeds chunk size, split it by sentences
        if para_tokens > CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', para_stripped)
            for sentence in sentences:
                sentence_tokens = estimate_tokens(sentence, model_name)
                
                if current_chunk_tokens + sentence_tokens <= CHUNK_SIZE:
                    current_chunk += sentence + " "
                    current_chunk_tokens += sentence_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_chunk_tokens = sentence_tokens
        # Normal case: add paragraph if it fits
        elif current_chunk_tokens + para_tokens <= CHUNK_SIZE:
            current_chunk += para_stripped + "\n\n"
            current_chunk_tokens += para_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para_stripped + "\n\n"
            current_chunk_tokens = para_tokens
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks


def truncate_to_token_limit(text: str, token_limit: int, model_name: Optional[str] = None) -> str:
    """
    Truncate text to fit within a token limit while preserving meaning.
    
    Args:
        text: Text to truncate
        token_limit: Maximum tokens allowed
        model_name: Optional model name for tokenizer
        
    Returns:
        Truncated text
    """
    if estimate_tokens(text, model_name) <= token_limit:
        return text
        
    # Try to truncate at paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text)
    result = ""
    current_tokens = 0
    
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
            
        para_tokens = estimate_tokens(para_stripped, model_name)
        
        if current_tokens + para_tokens <= token_limit:
            result += para_stripped + "\n\n"
            current_tokens += para_tokens
        else:
            # Try to include partial paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para_stripped)
            for sentence in sentences:
                sentence_tokens = estimate_tokens(sentence, model_name)
                if current_tokens + sentence_tokens <= token_limit:
                    result += sentence + " "
                    current_tokens += sentence_tokens
                else:
                    break
            break
    
    return result.strip() + "\n\n[content truncated to fit token limit]"


def trim_messages(messages: List[dict], max_tokens: int = DEFAULT_MAX_TOKENS) -> List[dict]:
    """
    Trim messages to fit within token budget, preserving the system and latest user message.
    
    Args:
        messages: List of message dicts with 'content' field
        max_tokens: Max token budget
        
    Returns:
        Trimmed message list
    """
    if not messages or len(messages) < 2:
        return messages

    system_msg = messages[0]           
    final_user_msg = messages[-1]      
    middle_msgs = messages[1:-1]        
    total_tokens = estimate_tokens(system_msg["content"]) + estimate_tokens(final_user_msg["content"])
    trimmed_middle = []

    for msg in middle_msgs:
        token_est = estimate_tokens(msg["content"])
        if total_tokens + token_est > max_tokens:
            break
        trimmed_middle.append(msg)
        total_tokens += token_est

    return [system_msg] + trimmed_middle + [final_user_msg]

