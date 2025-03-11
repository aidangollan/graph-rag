import logging
import asyncio
from langchain_core.prompts import ChatPromptTemplate

from utils.llm import get_llm
from objects.knowledge_graph import KnowledgeGraph
from utils.constants import GPT_4O_MINI, KG_SYSTEM_PROMPT, KG_USER_TEMPLATE
from services.embedding_service import EmbeddingService

def escape_template_variables(s):
    return s.replace("{", "{{").replace("}", "}}")

async def setup_llm(text: str, summary: str):
    extraction_llm = get_llm(GPT_4O_MINI).with_structured_output(KnowledgeGraph, include_raw=True)
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", KG_SYSTEM_PROMPT),
        ("human", KG_USER_TEMPLATE),
    ])
    
    return extraction_prompt | extraction_llm

class CustomKGBuilder:
    def __init__(self):
        """
        Initialize the knowledge graph builder with an embedding service.
        """
        self.embedding_service = EmbeddingService()
    
    async def create_knowledge_graph(self, text: str, summary: str) -> KnowledgeGraph:
        """
        Create a knowledge graph from text and summary, with embeddings for nodes.
        
        Args:
            text: The text to extract knowledge from
            summary: A summary of the text
            
        Returns:
            A knowledge graph with nodes and relationships
        """
        try:
            logging.info("Generating knowledge graph.")

            cleaned_text = escape_template_variables(text)
            cleaned_summary = escape_template_variables(summary)

            logging.info(f"Cleaned text: {cleaned_text[:100]}...")
            logging.info(f"Cleaned summary: {cleaned_summary[:100]}...")
            
            llm_chain = await setup_llm(text=cleaned_text, summary=cleaned_summary)

            try:
                result = await llm_chain.ainvoke({"text": cleaned_text, "summary": cleaned_summary})
                logging.info(f"LLM output received")
                
                # Parse the LLM output to get a knowledge graph
                knowledge_graph = KnowledgeGraph.parse_llm_output(result)
                logging.info(f"Parsed knowledge graph with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
                
                # Generate embeddings for all nodes in the knowledge graph
                if len(knowledge_graph.nodes) > 0:
                    logging.info(f"Generating embeddings for {len(knowledge_graph.nodes)} nodes")
                    knowledge_graph = await self.embedding_service.generate_embeddings_for_graph(knowledge_graph)
                    logging.info(f"Generated embeddings for knowledge graph")
                else:
                    logging.warning("No nodes found in knowledge graph, skipping embedding generation")

                return knowledge_graph

            except Exception as e:
                logging.exception(f"Error processing LLM output: {str(e)}")
                # Return an empty KnowledgeGraph if there's an error
                return KnowledgeGraph(nodes=[], relationships=[])

        except Exception as e:
            logging.exception(f"Error in create_knowledge_graph: {str(e)}")
            # Return an empty KnowledgeGraph if there's an error
            return KnowledgeGraph(nodes=[], relationships=[])