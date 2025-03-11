import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.pydantic_v1 import BaseModel, Field, validator

class Node(BaseModel):
    id: str = Field(description="Name or human-readable unique identifier. Must be all lowercase with spaces between words.")
    description: str = Field(description="Description of the node as would be read on a flashcard.")
    embedding_score: Optional[float] = Field(default=None, description="Similarity score for embedding-based searches.")

class Relationship(BaseModel):
    source: str = Field(description="Name or human-readable unique identifier of source node, must match a node in the nodes list. Must be all lowercase with spaces between words.")
    target: str = Field(description="Name or human-readable unique identifier of target node, must match a node in the nodes list. Must be all lowercase with spaces between words.")
    type: str = Field(description="The type of the relationship.")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="List of concept or entity nodes")
    relationships: List[Relationship] = Field(description="List of relationships between concepts or entities")

    @staticmethod
    def parse_llm_output(output) -> 'KnowledgeGraph':
        """
        Parse the output from the LLM and convert it to a KnowledgeGraph object.
        
        Args:
            output: Output from the LLM
            
        Returns:
            KnowledgeGraph object
        """
        try:
            logging.info(f"Parsing LLM output type: {type(output)}")
            
            # If output is already a KnowledgeGraph, return it
            if isinstance(output, KnowledgeGraph):
                logging.info("Output is already a KnowledgeGraph instance")
                return output
                
            # Check if output is a dictionary with 'parsed' key
            if isinstance(output, dict) and 'parsed' in output and output['parsed'] is not None:
                if isinstance(output['parsed'], KnowledgeGraph):
                    logging.info("Found KnowledgeGraph in output['parsed']")
                    return output['parsed']
            
            # Initialize empty knowledge graph
            knowledge_graph = KnowledgeGraph(nodes=[], relationships=[])
            
            # Extract data from tool_calls if available
            if isinstance(output, dict) and 'raw' in output and output['raw'] is not None:
                raw_output = output['raw']
                
                # Check if raw_output has tool_calls
                if hasattr(raw_output, 'additional_kwargs') and 'tool_calls' in raw_output.additional_kwargs:
                    tool_calls = raw_output.additional_kwargs['tool_calls']
                    if tool_calls and len(tool_calls) > 0:
                        # Extract arguments from the first tool call
                        arguments_str = tool_calls[0]['function']['arguments']
                        logging.info(f"Extracted arguments: {arguments_str[:100]}...")
                        
                        try:
                            # Parse the arguments JSON
                            arguments = json.loads(arguments_str)
                            
                            # Process nodes
                            if 'nodes' in arguments:
                                for node_data in arguments['nodes']:
                                    if 'id' in node_data and 'description' in node_data:
                                        node = Node(
                                            id=node_data['id'],
                                            description=node_data['description'],
                                            embedding_score=node_data.get('embedding_score')
                                        )
                                        knowledge_graph.nodes.append(node)
                            
                            # Process relationships
                            if 'relationships' in arguments:
                                for rel_data in arguments['relationships']:
                                    source = rel_data.get('source')
                                    target = rel_data.get('target')
                                    rel_type = rel_data.get('type')
                                    
                                    if source and target and rel_type:
                                        rel = Relationship(
                                            source=source,
                                            target=target,
                                            type=rel_type
                                        )
                                        knowledge_graph.relationships.append(rel)
                            
                            logging.info(f"Created knowledge graph with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
                            return knowledge_graph
                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing JSON arguments: {str(e)}")
                
                # If we couldn't extract from tool_calls, try to extract from content
                if hasattr(raw_output, 'content') and raw_output.content:
                    content = raw_output.content
                    # Try to find JSON in the content
                    try:
                        # Find JSON-like structure in the content
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = content[start_idx:end_idx]
                            arguments = json.loads(json_str)
                            
                            # Process nodes and relationships as above
                            if 'nodes' in arguments:
                                for node_data in arguments['nodes']:
                                    node = Node(
                                        id=node_data['id'],
                                        description=node_data['description'],
                                        embedding_score=node_data.get('embedding_score')
                                    )
                                    knowledge_graph.nodes.append(node)
                            
                            if 'relationships' in arguments:
                                for rel_data in arguments['relationships']:
                                    rel = Relationship(
                                        source=rel_data['source'],
                                        target=rel_data['target'],
                                        type=rel_data['type']
                                    )
                                    knowledge_graph.relationships.append(rel)
                            
                            logging.info(f"Created knowledge graph from content with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
                            return knowledge_graph
                    except Exception as e:
                        logging.error(f"Error extracting JSON from content: {str(e)}")
            
            # If we get here and the knowledge graph is still empty, check if output has 'tool_calls' directly
            if isinstance(output, dict) and 'tool_calls' in output and len(knowledge_graph.nodes) == 0:
                tool_calls = output['tool_calls']
                if tool_calls and len(tool_calls) > 0 and 'args' in tool_calls[0]:
                    args = tool_calls[0]['args']
                    
                    # Process nodes
                    if 'nodes' in args:
                        for node_data in args['nodes']:
                            node = Node(
                                id=node_data['id'],
                                description=node_data['description'],
                                embedding_score=node_data.get('embedding_score')
                            )
                            knowledge_graph.nodes.append(node)
                    
                    # Process relationships
                    if 'relationships' in args:
                        for rel_data in args['relationships']:
                            rel = Relationship(
                                source=rel_data['source'],
                                target=rel_data['target'],
                                type=rel_data['type']
                            )
                            knowledge_graph.relationships.append(rel)
                    
                    logging.info(f"Created knowledge graph from tool_calls args with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
                    return knowledge_graph
            
            # If we still have an empty knowledge graph, log an error
            if len(knowledge_graph.nodes) == 0:
                logging.error(f"Could not extract knowledge graph from output: {output}")
            
            return knowledge_graph
            
        except Exception as e:
            logging.error(f"Error in parse_llm_output: {str(e)}")
            # Return empty knowledge graph on error
            return KnowledgeGraph(nodes=[], relationships=[])