"""
Service for managing node embeddings and similarity searches.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from objects.knowledge_graph import Node, KnowledgeGraph, Relationship
from utils.embeddings import generate_embedding, calculate_similarity
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
        # Dictionary to temporarily store embeddings for nodes
        self.node_embeddings = {}
        
    def generate_node_embedding(self, node: Node) -> Node:
        """
        Generate an embedding for a node based on its description and store it in memory.
        
        Args:
            node: The node to generate an embedding for
            
        Returns:
            The node with embedding_score set to None (will be populated during similarity search)
        """
        try:
            # Generate the embedding and store it in our internal dictionary
            node_text = f"{node.id} {node.description}"
            embedding = generate_embedding(node_text)
            
            # Store the embedding in our dictionary using node.id as the key
            self.node_embeddings[node.id] = embedding
            
            return node
        except Exception as e:
            logging.error(f"Error generating node embedding: {str(e)}")
            return node
    
    def generate_embeddings_for_graph(self, knowledge_graph: KnowledgeGraph) -> KnowledgeGraph:
        """
        Generate embeddings for all nodes in a knowledge graph.
        
        Args:
            knowledge_graph: The knowledge graph to generate embeddings for
            
        Returns:
            The knowledge graph with embeddings generated for all nodes
        """
        try:
            # Clear the embeddings dictionary before processing a new graph
            self.node_embeddings = {}
            
            for i, node in enumerate(knowledge_graph.nodes):
                knowledge_graph.nodes[i] = self.generate_node_embedding(node)
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
                # Create a Node object
                node = Node(
                    id=node_data["id"],
                    description=node_data["description"],
                    embedding_score=None
                )
                
                # Get the node embedding from our internal dictionary
                node_embedding = self.node_embeddings.get(node.id)
                
                if node_embedding is None:
                    # If the node embedding is not in our dictionary, generate it on the fly
                    node_text = f"{node.id} {node.description}"
                    node_embedding = generate_embedding(node_text)
                
                # Calculate similarity score
                similarity_score = calculate_similarity(query_embedding, node_embedding)
                node.embedding_score = similarity_score
                
                nodes_with_scores.append({
                    "id": node.id,
                    "description": node.description,
                    "embedding_score": similarity_score
                })
            
            # Sort by similarity score in descending order
            sorted_nodes = sorted(nodes_with_scores, key=lambda x: x["embedding_score"], reverse=True)
            
            # Return top k results
            return sorted_nodes[:top_k]
        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}")
            return []
    
    def update_node_embeddings(self, document_id: str) -> Dict[str, Any]:
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
            
            # Generate embeddings for the knowledge graph
            knowledge_graph = self.generate_embeddings_for_graph(knowledge_graph)
            
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
