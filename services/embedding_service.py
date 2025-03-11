"""
Service for managing node embeddings and similarity searches.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from objects.knowledge_graph import Node, KnowledgeGraph, Relationship
from utils.embeddings import generate_embedding, calculate_similarity, generate_embeddings_concurrently
from services.neo4j_service import Neo4jService

class EmbeddingService:
    """
    Service for managing embeddings and similarity searches on knowledge graph nodes.
    """
    
    def __init__(self, neo4j_service: Optional[Neo4jService] = None):
        """
        Initialize the embedding service.
        
        Args:
            neo4j_service: Optional Neo4j service instance. If not provided, a new one will be created.
        """
        self.neo4j_service = neo4j_service or Neo4jService()
        
    async def generate_node_embeddings_concurrently(self, nodes: List[Node]) -> List[Node]:
        """
        Generate embeddings for multiple nodes concurrently and set their embedding_score fields.
        
        Args:
            nodes: List of nodes to generate embeddings for
            
        Returns:
            List of nodes with embedding_score fields set
        """
        try:
            # Prepare texts for embedding generation
            node_texts = [f"{node.id} {node.description}" for node in nodes]
            
            # Generate embeddings concurrently
            embeddings = await generate_embeddings_concurrently(node_texts)
            
            # Set embedding_score field on each node
            for i, node in enumerate(nodes):
                # Store the raw embedding vector as the embedding_score
                # This will be used for similarity calculations
                node.embedding_score = embeddings[i]
                
            return nodes
        except Exception as e:
            logging.error(f"Error generating node embeddings concurrently: {str(e)}")
            return nodes
    
    def generate_node_embedding(self, node: Node) -> Node:
        """
        Generate an embedding for a node based on its description and set its embedding_score field.
        
        Args:
            node: The node to generate an embedding for
            
        Returns:
            The node with embedding_score set
        """
        try:
            # Generate the embedding
            node_text = f"{node.id} {node.description}"
            embedding = generate_embedding(node_text)
            
            # Set the embedding_score field on the node
            node.embedding_score = embedding
            
            return node
        except Exception as e:
            logging.error(f"Error generating node embedding: {str(e)}")
            return node
    
    async def generate_embeddings_for_graph(self, knowledge_graph: KnowledgeGraph) -> KnowledgeGraph:
        """
        Generate embeddings for all nodes in a knowledge graph concurrently.
        
        Args:
            knowledge_graph: The knowledge graph to generate embeddings for
            
        Returns:
            The knowledge graph with embeddings generated for all nodes
        """
        try:
            if not knowledge_graph.nodes:
                logging.warning("Knowledge graph has no nodes to generate embeddings for")
                return knowledge_graph
                
            # Generate embeddings concurrently
            knowledge_graph.nodes = await self.generate_node_embeddings_concurrently(knowledge_graph.nodes)
            
            # Log success
            logging.info(f"Generated embeddings for {len(knowledge_graph.nodes)} nodes concurrently")
            
            return knowledge_graph
        except Exception as e:
            logging.error(f"Error generating embeddings for graph: {str(e)}")
            return knowledge_graph
    
    def similarity_search(self, query: str, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a similarity search on nodes in a document.
        
        Args:
            query: The search query
            document_id: The document ID to search within
            top_k: The number of top results to return
            
        Returns:
            List of nodes with similarity scores
        """
        try:
            # Get document graph from Neo4j
            document_graph = self.neo4j_service.get_document_graph(document_id)
            
            if document_graph.get("status") != "success":
                logging.error(f"Error retrieving document graph: {document_graph.get('error', 'Unknown error')}")
                return []
            
            # Generate embedding for the query
            query_embedding = generate_embedding(query)
            
            # Calculate similarity scores for each node
            nodes_with_scores = []
            for node_data in document_graph.get("nodes", []):
                # Get the embedding from the node data
                node_embedding = node_data.get("embedding_score", [])
                
                # If the node has no embedding, generate one on the fly
                if not node_embedding:
                    node_text = f"{node_data['id']} {node_data['description']}"
                    node_embedding = generate_embedding(node_text)
                
                # Calculate similarity score
                similarity_score = calculate_similarity(query_embedding, node_embedding)
                
                nodes_with_scores.append({
                    "id": node_data["id"],
                    "description": node_data["description"],
                    "similarity_score": similarity_score
                })
            
            # Sort by similarity score in descending order
            sorted_nodes = sorted(nodes_with_scores, key=lambda x: x["similarity_score"], reverse=True)
            
            # Return top k results
            return sorted_nodes[:top_k]
        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}")
            return []
    
    async def update_node_embeddings(self, document_id: str) -> Dict[str, Any]:
        """
        Update embeddings for all nodes in a document.
        
        Args:
            document_id: The document ID to update embeddings for
            
        Returns:
            Dictionary with information about the update
        """
        try:
            # Get document graph from Neo4j
            document_graph = self.neo4j_service.get_document_graph(document_id)
            
            if document_graph.get("status") != "success":
                logging.error(f"Error retrieving document graph: {document_graph.get('error', 'Unknown error')}")
                return {
                    "document_id": document_id,
                    "status": "error",
                    "error": document_graph.get("error", "Unknown error")
                }
            
            # Create a knowledge graph from the document graph
            nodes = []
            for node_data in document_graph.get("nodes", []):
                node = Node(
                    id=node_data["id"],
                    description=node_data["description"],
                    embedding_score=node_data.get("embedding_score")
                )
                nodes.append(node)
            
            relationships = []
            for rel_data in document_graph.get("relationships", []):
                relationship = Relationship(
                    source=rel_data["source"],
                    target=rel_data["target"],
                    type=rel_data["type"]
                )
                relationships.append(relationship)
            
            knowledge_graph = KnowledgeGraph(nodes=nodes, relationships=relationships)
            
            # Generate embeddings for the knowledge graph concurrently
            knowledge_graph = await self.generate_embeddings_for_graph(knowledge_graph)
            
            # Commit the updated knowledge graph to Neo4j
            result = self.neo4j_service.commit_knowledge_graph(knowledge_graph, document_id)
            
            return {
                "document_id": document_id,
                "nodes_updated": len(knowledge_graph.nodes),
                "status": result.get("status", "error")
            }
        except Exception as e:
            logging.error(f"Error updating node embeddings: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
