import os
import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase
from dotenv import load_dotenv
from objects.knowledge_graph import KnowledgeGraph, Node, Relationship
import numpy as np

# Load environment variables
load_dotenv()

class Neo4jService:
    """
    Service for interacting with Neo4j database.
    """
    
    def __init__(self):
        """
        Initialize the Neo4j service with connection details from environment variables.
        
        Environment variables required:
        - NEO4J_URI: URI for the Neo4j database
        - NEO4J_USERNAME: Username for Neo4j authentication
        - NEO4J_PASSWORD: Password for Neo4j authentication
        """
        try:
            self.uri = os.environ.get("NEO4J_URI")
            self.username = os.environ.get("NEO4J_USERNAME")
            self.password = os.environ.get("NEO4J_PASSWORD")
            
            if not all([self.uri, self.username, self.password]):
                logging.error("Missing Neo4j environment variables")
                raise ValueError("Missing Neo4j environment variables")
                
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            logging.info("Successfully connected to Neo4j database")
        except Exception as e:
            logging.error(f"Failed to initialize Neo4j connection: {str(e)}")
            raise
    
    def close(self):
        """
        Close the Neo4j driver connection.
        """
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def commit_knowledge_graph(self, knowledge_graph: KnowledgeGraph, document_id: str) -> Dict[str, Any]:
        """
        Commit a knowledge graph to the Neo4j database.
        
        Args:
            knowledge_graph: The knowledge graph to commit
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary with information about the committed graph
        """
        try:
            # Log the knowledge graph details for debugging
            logging.info(f"Committing knowledge graph with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
            
            if len(knowledge_graph.nodes) == 0:
                logging.warning("Knowledge graph has no nodes to commit")
                return {
                    "document_id": document_id,
                    "nodes_committed": 0,
                    "relationships_committed": 0,
                    "status": "warning",
                    "message": "Knowledge graph has no nodes to commit"
                }
            
            with self.driver.session() as session:
                # Add document node
                session.execute_write(self._create_document_node, document_id)
                
                # Add concept nodes and track successful creations
                successful_nodes = []
                for node in knowledge_graph.nodes:
                    try:
                        # Check if the node has an embedding score
                        if not hasattr(node, 'embedding_score') or node.embedding_score is None:
                            # Set a default embedding score of 0.0 if none exists
                            node.embedding_score = 0.0
                            logging.info(f"Setting default embedding score for node: {node.id}")
                        
                        session.execute_write(self._create_concept_node, node, document_id)
                        successful_nodes.append(node.id)
                        logging.info(f"Successfully created node: {node.id} with embedding score: {node.embedding_score}")
                    except Exception as e:
                        logging.error(f"Error creating node {node.id}: {str(e)}")
                
                # Add relationships for successfully created nodes
                successful_relationships = 0
                for relationship in knowledge_graph.relationships:
                    if relationship.source in successful_nodes and relationship.target in successful_nodes:
                        try:
                            session.execute_write(self._create_relationship, relationship, document_id)
                            successful_relationships += 1
                            logging.info(f"Successfully created relationship: {relationship.source} -{relationship.type}-> {relationship.target}")
                        except Exception as e:
                            logging.error(f"Error creating relationship {relationship.source} -{relationship.type}-> {relationship.target}: {str(e)}")
                
                # Get statistics
                node_count = len(successful_nodes)
                relationship_count = successful_relationships
                
                # Log success
                logging.info(f"Successfully committed knowledge graph to Neo4j: {node_count} nodes, {relationship_count} relationships")
                
                return {
                    "document_id": document_id,
                    "nodes_committed": node_count,
                    "relationships_committed": relationship_count,
                    "status": "success"
                }
        except Exception as e:
            logging.error(f"Error committing knowledge graph to Neo4j: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    def _create_document_node(tx, document_id: str):
        """
        Create a document node in the Neo4j database.
        
        Args:
            tx: Neo4j transaction
            document_id: Unique identifier for the document
        """
        query = (
            "MERGE (d:Document {id: $document_id}) "
            "RETURN d"
        )
        result = tx.run(query, document_id=document_id)
        # Consume the result to ensure the query is executed
        summary = result.consume()
        logging.info(f"Document node created: {summary.counters}")
    
    @staticmethod
    def _create_concept_node(tx, node: Node, document_id: str) -> None:
        """
        Create a concept node in Neo4j.
        
        Args:
            tx: Neo4j transaction
            node: Node to create
            document_id: Document ID to link the node to
        """
        # Convert embedding_score to a string if it's a list or array
        embedding_score = node.embedding_score
        if isinstance(embedding_score, (list, np.ndarray)):
            # Convert numpy array to list if needed
            if hasattr(embedding_score, 'tolist'):
                embedding_score = embedding_score.tolist()
                
            # Serialize the embedding vector to a string for storage
            embedding_score = str(embedding_score)
        
        # Create the node
        query = """
        MERGE (c:Concept {id: $id})
        SET c.description = $description,
            c.embedding_score = $embedding_score
        WITH c
        
        MATCH (d:Document {id: $document_id})
        MERGE (c)-[:BELONGS_TO]->(d)
        """
        
        tx.run(
            query,
            id=node.id,
            description=node.description,
            embedding_score=embedding_score,
            document_id=document_id
        )
    
    @staticmethod
    def _create_relationship(tx, relationship: Relationship, document_id: str):
        """
        Create a relationship between concept nodes in the Neo4j database.
        
        Args:
            tx: Neo4j transaction
            relationship: The relationship to create
            document_id: Unique identifier for the document
        """
        # Sanitize relationship type for Neo4j (replace spaces with underscores and uppercase)
        rel_type = relationship.type.upper().replace(" ", "_")
        
        try:
            # First ensure both nodes exist
            ensure_nodes_query = (
                "MERGE (source:Concept {id: $source_id}) "
                "MERGE (target:Concept {id: $target_id}) "
                "RETURN source, target"
            )
            
            tx.run(
                ensure_nodes_query,
                source_id=relationship.source,
                target_id=relationship.target
            )
            
            # Create the relationship with a standard type to avoid issues
            # Using RELATED_TO as the base relationship type with properties for the specific type
            query = (
                "MATCH (source:Concept {id: $source_id}), (target:Concept {id: $target_id}) "
                "MERGE (source)-[r:RELATED_TO {type: $rel_type, document_id: $document_id}]->(target) "
                "RETURN r"
            )
            
            result = tx.run(
                query, 
                source_id=relationship.source, 
                target_id=relationship.target, 
                rel_type=relationship.type,
                document_id=document_id
            )
            
            summary = result.consume()
            logging.info(f"Relationship created: {relationship.type}, {summary.counters}")
            
        except Exception as e:
            logging.error(f"Error creating relationship: {str(e)}")
            raise
    
    def get_document_graph(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a knowledge graph for a specific document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary containing nodes and relationships for the document
        """
        try:
            with self.driver.session() as session:
                # Check if document exists
                document_exists = session.execute_read(self._check_document_exists, document_id)
                
                if not document_exists:
                    return {
                        "document_id": document_id,
                        "status": "error",
                        "error": "Document not found"
                    }
                
                nodes = session.execute_read(self._get_document_nodes, document_id)
                relationships = session.execute_read(self._get_document_relationships, document_id)
                
                return {
                    "document_id": document_id,
                    "nodes": nodes,
                    "relationships": relationships,
                    "status": "success"
                }
        except Exception as e:
            logging.error(f"Error retrieving document graph: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    def _check_document_exists(tx, document_id: str) -> bool:
        """
        Check if a document exists in the database.
        
        Args:
            tx: Neo4j transaction
            document_id: Unique identifier for the document
            
        Returns:
            True if the document exists, False otherwise
        """
        query = "MATCH (d:Document {id: $document_id}) RETURN count(d) as count"
        result = tx.run(query, document_id=document_id)
        record = result.single()
        return record and record["count"] > 0
    
    @staticmethod
    def _get_document_nodes(tx, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all nodes for a document.
        
        Args:
            tx: Neo4j transaction
            document_id: Document ID
            
        Returns:
            List of nodes with their properties
        """
        query = (
            "MATCH (c:Concept)-[:BELONGS_TO]->(d:Document {id: $document_id}) "
            "RETURN c.id as id, c.description as description, c.embedding_score as embedding_score"
        )
        result = tx.run(query, document_id=document_id)
        
        nodes = []
        for record in result:
            # Get embedding score and try to parse it if it's a string representation of a list
            embedding_score = record.get("embedding_score")
            if embedding_score and isinstance(embedding_score, str) and embedding_score.startswith('[') and embedding_score.endswith(']'):
                try:
                    # Convert string representation of list back to actual list
                    import ast
                    embedding_score = ast.literal_eval(embedding_score)
                except (ValueError, SyntaxError) as e:
                    logging.warning(f"Failed to parse embedding score: {str(e)}")
            
            nodes.append({
                "id": record.get("id"),
                "description": record.get("description"),
                "embedding_score": embedding_score
            })
        
        return nodes
    
    @staticmethod
    def _get_document_relationships(tx, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all relationships between nodes associated with a document.
        
        Args:
            tx: Neo4j transaction
            document_id: Unique identifier for the document
            
        Returns:
            List of relationships with their properties
        """
        query = (
            "MATCH (source:Concept)-[r:RELATED_TO]->(target:Concept) "
            "WHERE r.document_id = $document_id "
            "RETURN source.id as source, target.id as target, r.type as type"
        )
        result = tx.run(query, document_id=document_id)
        return [{"source": record["source"], "target": record["target"], "type": record["type"]} for record in result]

    def get_connected_chunks(self, document_id: str, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get chunks connected to the given nodes.
        
        Args:
            document_id: The document ID to search within
            node_ids: List of node IDs to find connected chunks for
            
        Returns:
            List of connected chunks
        """
        try:
            if not node_ids:
                return []
            
            with self.driver.session() as session:
                # Query to get chunks that contain the specified concepts
                # Note: The relationship direction is from chunk to concept (CHUNK)-[:CONTAINS]->(CONCEPT)
                query = """
                MATCH (d:Document {id: $document_id})
                MATCH (c:Concept)-[:BELONGS_TO]->(d)
                WHERE c.id IN $node_ids
                MATCH (chunk:Chunk)-[:CONTAINS]->(c)
                RETURN DISTINCT chunk.id AS id, chunk.content AS content
                """
                
                result = session.run(
                    query,
                    document_id=document_id,
                    node_ids=node_ids
                )
                
                # Process results
                chunks = []
                for record in result:
                    if record["id"] and record["content"]:
                        chunks.append({
                            "id": record["id"],
                            "content": record["content"]
                        })
                
                # If no chunks were found, use the node descriptions as context
                if not chunks:
                    logging.info("No chunks found, using node descriptions as context")
                    
                    # Get node descriptions
                    node_query = """
                    MATCH (d:Document {id: $document_id})
                    MATCH (c:Concept)-[:BELONGS_TO]->(d)
                    WHERE c.id IN $node_ids
                    RETURN c.id AS id, c.description AS description
                    """
                    
                    node_result = session.run(
                        node_query,
                        document_id=document_id,
                        node_ids=node_ids
                    )
                    
                    for record in node_result:
                        chunks.append({
                            "id": record["id"] + "_context",
                            "content": record["description"]
                        })
                
                return chunks
        except Exception as e:
            logging.error(f"Error getting connected chunks: {str(e)}")
            return []

    def get_all_document_ids(self) -> List[str]:
        """
        Get all document IDs stored in the Neo4j database.
        
        Returns:
            List of document IDs
        """
        try:
            with self.driver.session() as session:
                # Query to get all document IDs
                query = """
                MATCH (d:Document)
                RETURN d.id AS document_id
                ORDER BY d.id
                """
                
                result = session.run(query)
                
                # Process results
                document_ids = [record["document_id"] for record in result]
                
                return document_ids
        except Exception as e:
            logging.error(f"Error getting all document IDs: {str(e)}")
            return []

    @staticmethod
    def _create_chunk_node(tx, chunk_id: str, chunk_content: str, embedding_score, document_id: str) -> None:
        """
        Create a chunk node in Neo4j.
        
        Args:
            tx: Neo4j transaction
            chunk_id: Unique identifier for the chunk
            chunk_content: Text content of the chunk
            embedding_score: Embedding vector for the chunk
            document_id: Document ID to link the chunk to
        """
        # Convert embedding_score to a string if it's a list or array
        if isinstance(embedding_score, (list, np.ndarray)):
            # Convert numpy array to list if needed
            if hasattr(embedding_score, 'tolist'):
                embedding_score = embedding_score.tolist()
                
            # Serialize the embedding vector to a string for storage
            embedding_score = str(embedding_score)
        
        # Create the chunk node
        query = """
        MERGE (chunk:Chunk {id: $chunk_id})
        SET chunk.content = $content,
            chunk.embedding_score = $embedding_score
        WITH chunk
        
        MATCH (d:Document {id: $document_id})
        MERGE (chunk)-[:BELONGS_TO]->(d)
        """
        
        tx.run(
            query,
            chunk_id=chunk_id,
            content=chunk_content,
            embedding_score=embedding_score,
            document_id=document_id
        )
    
    @staticmethod
    def _create_chunk_concept_relationship(tx, chunk_id: str, concept_id: str):
        """
        Create a relationship between a chunk and a concept node in Neo4j.
        
        Args:
            tx: Neo4j transaction
            chunk_id: ID of the chunk node
            concept_id: ID of the concept node
        """
        query = """
        MATCH (chunk:Chunk {id: $chunk_id})
        MATCH (concept:Concept {id: $concept_id})
        MERGE (chunk)-[:CONTAINS]->(concept)
        """
        
        tx.run(
            query,
            chunk_id=chunk_id,
            concept_id=concept_id
        )
    
    def commit_knowledge_graph_with_chunks(self, knowledge_graph, document_id: str, chunks: List[str]) -> Dict[str, Any]:
        """
        Commit a knowledge graph with its chunks to the Neo4j database.
        
        Args:
            knowledge_graph: The knowledge graph to commit
            document_id: Unique identifier for the document
            chunks: List of text chunks from the document
            
        Returns:
            Dictionary with the result of the operation
        """
        try:
            with self.driver.session() as session:
                # Create document node
                session.execute_write(self._create_document_node, document_id)
                
                # Generate embeddings for chunks
                from utils.embeddings import generate_embedding
                import asyncio
                
                # Generate embeddings for each chunk
                chunk_embeddings = []
                for chunk in chunks:
                    embedding = generate_embedding(chunk)
                    chunk_embeddings.append(embedding)
                
                # Create chunk nodes
                chunk_ids = []
                for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_id = f"{document_id}_chunk_{i}"
                    chunk_ids.append(chunk_id)
                    session.execute_write(
                        self._create_chunk_node, 
                        chunk_id, 
                        chunk, 
                        embedding, 
                        document_id
                    )
                
                # Track successful operations
                successful_nodes = []
                successful_relationships = 0
                chunk_concept_relationships = 0
                
                # Create concept nodes
                for node in knowledge_graph.nodes:
                    try:
                        session.execute_write(self._create_concept_node, node, document_id)
                        successful_nodes.append(node.id)
                    except Exception as e:
                        logging.error(f"Error creating node {node.id}: {str(e)}")
                
                # Create relationships between concepts
                for relationship in knowledge_graph.relationships:
                    try:
                        session.execute_write(
                            self._create_relationship, 
                            relationship, 
                            document_id
                        )
                        successful_relationships += 1
                    except Exception as e:
                        logging.error(f"Error creating relationship {relationship.source} -{relationship.type}-> {relationship.target}: {str(e)}")
                
                # Create relationships between chunks and concepts
                # For each chunk, find which concepts are mentioned in it
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk_ids[i]
                    chunk_lower = chunk.lower()
                    
                    for node in knowledge_graph.nodes:
                        # Check if the node description is in the chunk
                        # This is a simple heuristic - we could use more sophisticated methods
                        if node.description.lower() in chunk_lower:
                            try:
                                # Create a relationship from the chunk to the concept
                                session.execute_write(
                                    self._create_chunk_concept_relationship,
                                    chunk_id,
                                    node.id
                                )
                                chunk_concept_relationships += 1
                            except Exception as e:
                                logging.error(f"Error creating chunk-concept relationship {chunk_id} -CONTAINS-> {node.id}: {str(e)}")
                
                # Get statistics
                node_count = len(successful_nodes)
                relationship_count = successful_relationships
                
                # Log success
                logging.info(f"Successfully committed knowledge graph with chunks to Neo4j: {node_count} nodes, {len(chunk_ids)} chunks, {relationship_count} concept relationships, {chunk_concept_relationships} chunk-concept relationships")
                
                return {
                    "document_id": document_id,
                    "nodes_committed": node_count,
                    "chunks_committed": len(chunk_ids),
                    "relationships_committed": relationship_count,
                    "chunk_concept_relationships": chunk_concept_relationships,
                    "status": "success"
                }
        except Exception as e:
            logging.error(f"Error committing knowledge graph with chunks to Neo4j: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }
