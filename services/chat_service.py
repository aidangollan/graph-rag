"""
Service for handling chat functionality with knowledge graph RAG.
"""
import logging
import re
from typing import Dict, Any, List, Tuple, Set
import asyncio

from objects.knowledge_graph import Node
from services.neo4j_service import Neo4jService
from services.embedding_service import EmbeddingService
from utils.llm import get_llm
from utils.constants import GPT_4O_MINI, GPT_35_TURBO_MODEL
from langchain_core.prompts import ChatPromptTemplate

class ChatService:
    """
    Service for handling chat functionality with knowledge graph RAG.
    """
    
    def __init__(self):
        """
        Initialize the chat service with Neo4j and embedding services.
        """
        self.neo4j_service = Neo4jService()
        self.embedding_service = EmbeddingService(self.neo4j_service)
    
    async def extract_proper_nouns(self, text: str) -> List[str]:
        """
        Extract proper nouns from the given text using OpenAI API.
        
        Args:
            text: The text to extract proper nouns from
            
        Returns:
            List of proper nouns
        """
        try:
            # Use OpenAI to extract proper nouns
            llm = get_llm(GPT_35_TURBO_MODEL)
            
            system_prompt = """
            You are an entity extraction expert. Your task is to extract all proper nouns and important entities from the given text.
            Return only a Python list of lowercase strings containing the extracted entities. 
            Include names, places, organizations, technical terms, and other important entities.
            Format your response as a valid Python list, e.g. ["entity1", "entity2", "entity3"]
            """
            
            user_prompt = f"Extract all proper nouns and important entities from this text: {text}"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            chain = prompt | llm
            response = await chain.ainvoke({})
            
            # Parse the response to extract the list
            response_text = response.content.strip()
            
            # Handle different formats of responses
            if response_text.startswith('[') and response_text.endswith(']'):
                # Try to parse as a Python list
                try:
                    # Use a safer approach than eval
                    response_text = response_text.replace("'", '"')  # Replace single quotes with double quotes for JSON parsing
                    import json
                    entities = json.loads(response_text)
                    return entities
                except Exception as e:
                    logging.error(f"Error parsing entity list: {str(e)}")
                    # Fall back to regex extraction
                    entities = re.findall(r'"([^"]+)"', response_text)
                    if not entities:
                        entities = re.findall(r"'([^']+)'", response_text)
                    return entities
            else:
                # If not in list format, extract words that might be entities
                words = re.findall(r'\b[A-Za-z]+\b', response_text)
                return [word.lower() for word in words]
                
        except Exception as e:
            logging.error(f"Error extracting proper nouns: {str(e)}")
            # Simple fallback using regex for capitalized words
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            return [word.lower() for word in words]
    
    async def query_nodes_by_proper_nouns(self, document_id: str, proper_nouns: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query nodes in the graph that match the given proper nouns.
        
        Args:
            document_id: The document ID to search within
            proper_nouns: List of proper nouns to search for
            top_k: Number of top results to return per proper noun
            
        Returns:
            List of nodes with similarity scores
        """
        if not proper_nouns:
            return []
        
        all_results = []
        
        # Query for each proper noun
        for noun in proper_nouns:
            results = self.embedding_service.similarity_search(noun, document_id, top_k)
            all_results.extend(results)
        
        # Remove duplicates based on node ID
        unique_results = {}
        for result in all_results:
            node_id = result["id"]
            if node_id not in unique_results or result["similarity_score"] > unique_results[node_id]["similarity_score"]:
                unique_results[node_id] = result
        
        # Convert back to list and sort by similarity score
        results_list = list(unique_results.values())
        results_list.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results_list
    
    def get_connected_chunks(self, document_id: str, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get chunks connected to the given nodes.
        
        Args:
            document_id: The document ID to search within
            node_ids: List of node IDs to find connected chunks for
            
        Returns:
            List of connected chunks
        """
        if not node_ids:
            return []
        
        # Get connected chunks from Neo4j
        connected_chunks = self.neo4j_service.get_connected_chunks(document_id, node_ids)
        
        return connected_chunks
    
    async def generate_response(self, query: str, context_nodes: List[Dict[str, Any]], context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response to the query using the provided context.
        
        Args:
            query: The user's query
            context_nodes: List of context nodes
            context_chunks: List of context chunks
            
        Returns:
            Generated response
        """
        try:
            # Prepare context from nodes and chunks
            nodes_context = "\n".join([f"Node: {node['id']}\nDescription: {node['description']}" for node in context_nodes])
            chunks_context = "\n".join([f"Chunk: {chunk['id']}\nContent: {chunk['content']}" for chunk in context_chunks])
            
            combined_context = f"Context Nodes:\n{nodes_context}\n\nContext Chunks:\n{chunks_context}"
            
            # Create system prompt
            system_prompt = """
            You are a helpful AI assistant with access to a knowledge graph about documents.
            Answer the user's question based on the provided context information.
            If the context doesn't contain enough information to answer the question, say so.
            Do not make up information that is not supported by the context.
            """
            
            # Create user prompt
            user_prompt = f"""
            Question: {query}
            
            Here is the context information from the knowledge graph:
            {combined_context}
            
            Please provide a detailed and accurate answer based on this context.
            """
            
            # Get LLM
            llm = get_llm(GPT_4O_MINI)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            # Generate response
            chain = prompt | llm
            response = await chain.ainvoke({})
            
            return response.content
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, but I encountered an error while generating a response: {str(e)}"
    
    async def process_chat_query(self, document_id: str, query: str) -> Dict[str, Any]:
        """
        Process a chat query using the knowledge graph.
        
        Args:
            document_id: The document ID to search within
            query: The user's query
            
        Returns:
            Dictionary with the response, used context nodes, and used context chunks
        """
        try:
            # Extract proper nouns from the query
            proper_nouns = await self.extract_proper_nouns(query)
            logging.info(f"Extracted proper nouns: {proper_nouns}")
            
            # Query nodes by proper nouns
            context_nodes = await self.query_nodes_by_proper_nouns(document_id, proper_nouns)
            logging.info(f"Found {len(context_nodes)} context nodes")
            
            # Get node IDs
            node_ids = [node["id"] for node in context_nodes]
            
            # Get connected chunks
            context_chunks = self.get_connected_chunks(document_id, node_ids)
            logging.info(f"Found {len(context_chunks)} connected chunks")
            
            # Extract chunk IDs
            chunk_ids = [chunk["id"] for chunk in context_chunks]
            logging.info(f"Using chunk IDs: {chunk_ids}")
            
            # Generate response
            response_text = await self.generate_response(query, context_nodes, context_chunks)
            
            # Return response with context information
            return {
                "document_id": document_id,
                "query": query,
                "response": response_text,
                "context_node_ids": node_ids,
                "context_chunk_ids": chunk_ids
            }
        except Exception as e:
            logging.error(f"Error processing chat query: {str(e)}")
            return {
                "document_id": document_id,
                "query": query,
                "response": f"I'm sorry, but I encountered an error while processing your query: {str(e)}",
                "context_node_ids": [],
                "context_chunk_ids": [],
                "error": str(e)
            }
