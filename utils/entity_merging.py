"""
Utility functions for merging similar entities in knowledge graphs.
"""
import logging
import json
from typing import List, Dict, Tuple, Set, Any
import re
from difflib import SequenceMatcher
from objects.knowledge_graph import KnowledgeGraph, Node, Relationship
from utils.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from utils.constants import (
    GPT_4O_MODEL,
    ENTITY_SIMILARITY_THRESHOLD,
    ENTITY_STOP_WORDS,
    MIN_WORD_LENGTH_FOR_PREFIX_CHECK,
    ENTITY_MERGING_SYSTEM_PROMPT,
    ENTITY_MERGING_USER_TEMPLATE
)

def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name by converting to lowercase and removing extra spaces.
    
    Args:
        name: Entity name to normalize
        
    Returns:
        Normalized entity name
    """
    return re.sub(r'\s+', ' ', name.lower().strip())

def calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two entity names using SequenceMatcher.
    
    Args:
        name1: First entity name
        name2: Second entity name
        
    Returns:
        Similarity score between 0 and 1
    """
    return SequenceMatcher(None, name1, name2).ratio()

def are_names_semantically_similar(name1: str, name2: str) -> bool:
    """
    Check if two names are semantically similar by looking for common significant words.
    
    Args:
        name1: First entity name
        name2: Second entity name
        
    Returns:
        True if names are semantically similar, False otherwise
    """
    # Normalize names
    norm1 = normalize_entity_name(name1)
    norm2 = normalize_entity_name(name2)
    
    # If one is a substring of the other, they're likely related
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    # Split into words and remove common stop words
    words1 = [w for w in norm1.split() if w not in ENTITY_STOP_WORDS]
    words2 = [w for w in norm2.split() if w not in ENTITY_STOP_WORDS]
    
    # If either set is empty after removing stop words, return False
    if not words1 or not words2:
        return False
    
    # Calculate word overlap
    common_words = set(words1).intersection(set(words2))
    
    # If there's at least one significant common word, consider them similar
    if common_words:
        return True
    
    # Check for common prefixes/suffixes in words
    for w1 in words1:
        for w2 in words2:
            # If one word is at least 4 chars and is a prefix of the other
            if len(w1) >= MIN_WORD_LENGTH_FOR_PREFIX_CHECK and w2.startswith(w1):
                return True
            if len(w2) >= MIN_WORD_LENGTH_FOR_PREFIX_CHECK and w1.startswith(w2):
                return True
    
    return False

def find_similar_entity_groups(knowledge_graph: KnowledgeGraph, similarity_threshold: float = ENTITY_SIMILARITY_THRESHOLD) -> List[List[int]]:
    """
    Find groups of similar entities in a knowledge graph based on name similarity.
    
    Args:
        knowledge_graph: Knowledge graph to process
        similarity_threshold: Threshold for name similarity (default: 0.85)
        
    Returns:
        List of groups, where each group is a list of node indices
    """
    if not knowledge_graph.nodes:
        return []
    
    # Create a mapping of node IDs to their indices
    node_indices = {node.id: i for i, node in enumerate(knowledge_graph.nodes)}
    
    # Create a graph where nodes are connected if they're similar
    similarity_graph = {}
    
    # Initialize the graph
    for i in range(len(knowledge_graph.nodes)):
        similarity_graph[i] = set()
    
    # Connect similar nodes
    for i, node1 in enumerate(knowledge_graph.nodes):
        for j, node2 in enumerate(knowledge_graph.nodes):
            if i >= j:  # Skip self-connections and duplicates
                continue
            
            # Calculate string similarity
            name1 = normalize_entity_name(node1.id)
            name2 = normalize_entity_name(node2.id)
            
            string_similarity = calculate_name_similarity(name1, name2)
            
            # Check for semantic similarity
            semantic_similarity = are_names_semantically_similar(name1, name2)
            
            # Connect nodes if they're similar by either measure
            if string_similarity >= similarity_threshold or semantic_similarity:
                similarity_graph[i].add(j)
                similarity_graph[j].add(i)
    
    # Find connected components (groups of similar entities)
    visited = set()
    groups = []
    
    for i in range(len(knowledge_graph.nodes)):
        if i in visited:
            continue
        
        # BFS to find all connected nodes
        group = []
        queue = [i]
        visited.add(i)
        
        while queue:
            node = queue.pop(0)
            group.append(node)
            
            for neighbor in similarity_graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups

async def llm_decide_entity_merges(knowledge_graph: KnowledgeGraph, potential_groups: List[List[int]]) -> List[Dict[str, Any]]:
    """
    Use LLM to decide which entities should be merged and how.
    
    Args:
        knowledge_graph: Knowledge graph to process
        potential_groups: List of groups of similar entity indices
        
    Returns:
        List of merge decisions, each containing the final entity ID, description, and indices to merge
    """
    if not potential_groups:
        return []
    
    # Format the entity groups for the prompt
    entity_groups_text = ""
    for i, group in enumerate(potential_groups):
        entity_groups_text += f"Group {i}:\n"
        for idx in group:
            node = knowledge_graph.nodes[idx]
            entity_groups_text += f"  Index {idx}: ID='{node.id}', Description='{node.description}'\n"
        entity_groups_text += "\n"

    # Create the LLM chain
    llm = get_llm(GPT_4O_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_MERGING_SYSTEM_PROMPT),
        ("human", ENTITY_MERGING_USER_TEMPLATE)
    ])
    
    chain = prompt | llm
    
    # Invoke the LLM
    try:
        response = await chain.ainvoke({"entity_groups": entity_groups_text})
        
        # Extract JSON from the response
        json_start = response.content.find('```json') + 7
        json_end = response.content.find('```', json_start)
        
        if json_start >= 7 and json_end > json_start:
            json_str = response.content[json_start:json_end].strip()
            merge_decisions = json.loads(json_str)
            
            # Validate the merge decisions
            valid_decisions = []
            for decision in merge_decisions:
                if (isinstance(decision, dict) and 
                    'should_merge' in decision and 
                    decision['should_merge'] and
                    'final_id' in decision and
                    'final_description' in decision and
                    'indices' in decision):
                    valid_decisions.append(decision)
            
            return valid_decisions
        else:
            logging.error("Could not extract JSON from LLM response")
            return []
    
    except Exception as e:
        logging.error(f"Error in LLM entity merge decision: {str(e)}")
        return []

async def merge_similar_entities(knowledge_graph: KnowledgeGraph, similarity_threshold: float = ENTITY_SIMILARITY_THRESHOLD) -> KnowledgeGraph:
    """
    Merge similar entities in a knowledge graph using LLM decisions.
    
    Args:
        knowledge_graph: Knowledge graph to process
        similarity_threshold: Threshold for name similarity (default: 0.85)
        
    Returns:
        Knowledge graph with merged entities
    """
    if not knowledge_graph.nodes:
        return knowledge_graph
    
    logging.info(f"Starting entity merging with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
    
    # Find potential groups of similar entities
    potential_groups = find_similar_entity_groups(knowledge_graph, similarity_threshold)
    
    if not potential_groups:
        logging.info("No similar entities found to merge")
        return knowledge_graph
    
    logging.info(f"Found {len(potential_groups)} potential groups of similar entities")
    
    # Use LLM to decide which entities to merge
    merge_decisions = await llm_decide_entity_merges(knowledge_graph, potential_groups)
    
    if not merge_decisions:
        logging.info("LLM decided not to merge any entities")
        return knowledge_graph
    
    logging.info(f"LLM decided to merge {len(merge_decisions)} groups of entities")
    
    # Create a new list of nodes with merged entities
    new_nodes = []
    id_mapping = {}  # Maps old node IDs to new node IDs
    
    # Track which indices will be merged
    indices_to_merge = set()
    for decision in merge_decisions:
        indices_to_merge.update(decision['indices'])
    
    # First, add all nodes that aren't part of any merge
    for i, node in enumerate(knowledge_graph.nodes):
        if i not in indices_to_merge:
            new_nodes.append(node)
            id_mapping[node.id] = node.id
    
    # Then, process each merge decision to create merged nodes
    for decision in merge_decisions:
        # Create the merged node with LLM-decided ID and description
        merged_node = Node(
            id=decision['final_id'],
            description=decision['final_description'],
            embedding_score=None  # Will need to be recalculated
        )
        
        new_nodes.append(merged_node)
        
        # Update the ID mapping for all nodes in this merge
        for idx in decision['indices']:
            old_id = knowledge_graph.nodes[idx].id
            id_mapping[old_id] = decision['final_id']
    
    # Create new relationships with updated node IDs
    new_relationships = []
    seen_relationships = set()  # To avoid duplicate relationships
    
    for rel in knowledge_graph.relationships:
        # Map the source and target to their new IDs
        new_source = id_mapping.get(rel.source, rel.source)
        new_target = id_mapping.get(rel.target, rel.target)
        
        # Create a key to identify this relationship
        rel_key = (new_source, new_target, rel.type)
        
        # Only add this relationship if we haven't seen it before
        if rel_key not in seen_relationships:
            new_rel = Relationship(
                source=new_source,
                target=new_target,
                type=rel.type
            )
            new_relationships.append(new_rel)
            seen_relationships.add(rel_key)
    
    # Create the new knowledge graph
    merged_graph = KnowledgeGraph(
        nodes=new_nodes,
        relationships=new_relationships
    )
    
    logging.info(f"After merging, knowledge graph has {len(merged_graph.nodes)} nodes and {len(merged_graph.relationships)} relationships")
    
    return merged_graph
