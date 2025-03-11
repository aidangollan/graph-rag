"""
Utility functions for generating and managing embeddings.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np

# Import embedding models
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

def get_embedding_model() -> Embeddings:
    """
    Get the embedding model based on environment configuration.
    
    Returns:
        An instance of an embedding model
    """
    # Default to OpenAI embeddings
    try:
        # Check if OpenAI API key is available
        api_key = os.environ.get("OPENAI_KEY")
        if not api_key:
            logging.warning("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")
            
        # Initialize OpenAI embeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",  # Using the latest embedding model
            openai_api_key=api_key
        )
    except Exception as e:
        logging.error(f"Error initializing embedding model: {str(e)}")
        raise

def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        A list of floats representing the embedding vector
    """
    try:
        model = get_embedding_model()
        embedding = model.embed_query(text)
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        return []

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate the cosine similarity between two embedding vectors.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    except Exception as e:
        logging.error(f"Error calculating similarity: {str(e)}")
        return 0.0
