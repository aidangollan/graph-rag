"""
Utility functions for chunking text into smaller segments.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
import tiktoken
from utils.constants import GPT_4O_MINI, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_MAX_CONCURRENCY

def count_tokens(text: str, model: str = GPT_4O_MINI) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model to use for tokenization
        
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        # Fallback: estimate tokens as words / 0.75 (rough approximation)
        return int(len(text.split()) / 0.75)

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP, model: str = GPT_4O_MINI) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: Target number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        model: The model to use for tokenization
        
    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        chunks = []
        
        # If text is smaller than chunk_size, return it as a single chunk
        if len(tokens) <= chunk_size:
            return [text]
        
        # Split into chunks with overlap
        i = 0
        while i < len(tokens):
            # Get chunk_size tokens (or remaining tokens if less)
            chunk_end = min(i + chunk_size, len(tokens))
            chunk_tokens = tokens[i:chunk_end]
            
            # Decode tokens back to text
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move to next chunk with overlap
            i += (chunk_size - overlap)
            
            # Ensure we don't get stuck in a loop if overlap >= chunk_size
            if i >= len(tokens) or i <= 0:
                break
                
        return chunks
    except Exception as e:
        logging.error(f"Error chunking text: {str(e)}")
        
        # Fallback: split by paragraphs and then combine
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk_size (estimated)
            if len(current_chunk.split()) + len(paragraph.split()) > chunk_size * 0.75:
                if current_chunk:  # Don't add empty chunks
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

async def process_chunks_concurrently(chunks: List[str], processor_func, max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> List[Any]:
    """
    Process text chunks concurrently using the provided processor function.
    
    Args:
        chunks: List of text chunks to process
        processor_func: Async function that processes a single chunk
        max_concurrency: Maximum number of chunks to process concurrently
        
    Returns:
        List of results from processing each chunk
    """
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(chunk):
        async with semaphore:
            return await processor_func(chunk)
    
    # Create tasks for all chunks
    tasks = [process_with_semaphore(chunk) for chunk in chunks]
    
    # Execute all tasks concurrently and gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = [result for result in results if not isinstance(result, Exception)]
    
    # Log any exceptions
    exceptions = [result for result in results if isinstance(result, Exception)]
    for exception in exceptions:
        logging.error(f"Error processing chunk: {str(exception)}")
    
    return valid_results
