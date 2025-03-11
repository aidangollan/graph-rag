import PyPDF2
import io
import uuid
import logging
import asyncio
from typing import Tuple, Dict, Any, List

from services.kg_service import CustomKGBuilder
from services.neo4j_service import Neo4jService
from utils.llm import get_llm
from utils.constants import GPT_4O_MINI
from utils.text_chunking import chunk_text, process_chunks_concurrently
from langchain_core.prompts import ChatPromptTemplate
from objects.knowledge_graph import KnowledgeGraph, Node, Relationship

class PDFService:
    """
    Service for processing PDF documents.
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> Tuple[str, int]:
        """
        Extract text from a PDF document.
        
        Args:
            pdf_content: Raw bytes of the PDF file
            
        Returns:
            Tuple containing:
                - Extracted text from the PDF
                - Number of pages in the PDF
                
        Raises:
            Exception: If there's an error processing the PDF
        """
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Get the number of pages
        num_pages = len(pdf_reader.pages)
        
        # Extract text from all pages
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
            
        return text, num_pages
    
    @staticmethod
    async def generate_summary(text: str) -> str:
        """
        Generate a summary of the PDF text using LLM.
        
        Args:
            text: Text extracted from the PDF
            
        Returns:
            Summary of the text
        """
        try:
            logging.info("Generating summary of PDF text")
            
            # Create a prompt for summarization
            system_prompt = """
            You are a professional summarizer. Your task is to create a concise and comprehensive summary 
            of the provided text. Focus on capturing the main ideas, key concepts, and important details.
            The summary should be approximately 15% of the original text length, but no more than 1000 words.
            """
            
            user_template = """
            Please summarize the following text:
            
            {text}
            """
            
            # Get LLM
            summary_llm = get_llm(GPT_4O_MINI)
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_template),
            ])
            
            # Generate summary
            chain = summary_prompt | summary_llm
            summary = await chain.ainvoke({"text": text})
            
            return summary.content
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            # Return a short excerpt of the text if summarization fails
            return text[:500] + "..." if len(text) > 500 else text
    
    @staticmethod
    async def process_chunk(chunk: str, kg_builder: CustomKGBuilder) -> KnowledgeGraph:
        """
        Process a single text chunk to generate a knowledge graph.
        
        Args:
            chunk: Text chunk to process
            kg_builder: Knowledge graph builder instance
            
        Returns:
            Knowledge graph for the chunk
        """
        try:
            logging.info(f"Processing chunk of size {len(chunk)} characters")
            
            # Generate a mini-summary for this chunk to help with context
            chunk_summary = await PDFService.generate_summary(chunk)
            
            # Generate knowledge graph for this chunk
            knowledge_graph = await kg_builder.create_knowledge_graph(chunk, chunk_summary)
            
            logging.info(f"Generated knowledge graph for chunk with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
            
            return knowledge_graph
        except Exception as e:
            logging.error(f"Error processing chunk: {str(e)}")
            return KnowledgeGraph(nodes=[], relationships=[])
    
    @staticmethod
    def merge_knowledge_graphs(graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Merge multiple knowledge graphs into a single graph.
        
        Args:
            graphs: List of knowledge graphs to merge
            
        Returns:
            Merged knowledge graph
        """
        merged_nodes = {}
        merged_relationships = set()
        
        # Merge nodes (use node ID as key to avoid duplicates)
        for graph in graphs:
            for node in graph.nodes:
                if node.id in merged_nodes:
                    # If node already exists, use the longer description
                    if len(node.description) > len(merged_nodes[node.id].description):
                        merged_nodes[node.id] = node
                else:
                    merged_nodes[node.id] = node
        
        # Merge relationships (use tuple of source, target, type as key to avoid duplicates)
        for graph in graphs:
            for rel in graph.relationships:
                rel_key = (rel.source, rel.target, rel.type)
                merged_relationships.add(rel_key)
        
        # Convert back to lists
        nodes_list = list(merged_nodes.values())
        relationships_list = [
            Relationship(source=src, target=tgt, type=rel_type)
            for src, tgt, rel_type in merged_relationships
        ]
        
        return KnowledgeGraph(nodes=nodes_list, relationships=relationships_list)
    
    @staticmethod
    async def process_pdf_with_knowledge_graph(pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a PDF document, extract text, generate knowledge graph, and store in Neo4j.
        Uses chunking and concurrent processing for improved performance.
        
        Args:
            pdf_content: Raw bytes of the PDF file
            filename: Name of the PDF file
            
        Returns:
            Dictionary with the processing results
        """
        try:
            # Extract text from PDF
            text, page_count = PDFService.extract_text_from_pdf(pdf_content)
            
            # Generate overall summary
            summary = await PDFService.generate_summary(text)
            
            # Generate document ID
            document_id = f"{filename.replace('.pdf', '')}_{uuid.uuid4().hex[:8]}"
            
            # Chunk the text into 250 token segments
            chunks = chunk_text(text, chunk_size=250, overlap=50)
            logging.info(f"Split document into {len(chunks)} chunks of approximately 250 tokens each")
            
            # Create knowledge graph builder
            kg_builder = CustomKGBuilder()
            
            # Process chunks concurrently with a processor function
            chunk_graphs = await process_chunks_concurrently(
                chunks=chunks,
                processor_func=lambda chunk: PDFService.process_chunk(chunk, kg_builder),
                max_concurrency=5  # Limit concurrency to avoid overwhelming the system
            )
            
            # Merge the knowledge graphs from all chunks
            knowledge_graph = PDFService.merge_knowledge_graphs(chunk_graphs)
            logging.info(f"Merged knowledge graph has {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
            
            # Merge similar entities across chunks
            from utils.entity_merging import merge_similar_entities
            knowledge_graph = await merge_similar_entities(knowledge_graph)
            logging.info(f"After entity merging, knowledge graph has {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.relationships)} relationships")
            
            # Store in Neo4j
            neo4j_service = Neo4jService()
            db_result = neo4j_service.commit_knowledge_graph(knowledge_graph, document_id)
            neo4j_service.close()
            
            # Format response
            response = PDFService.format_response(
                filename=filename,
                text=text,
                page_count=page_count,
                summary=summary,
                knowledge_graph=knowledge_graph,
                document_id=document_id,
                db_result=db_result,
                chunks_count=len(chunks)
            )
            
            return response
        except Exception as e:
            logging.error(f"Error processing PDF with knowledge graph: {str(e)}")
            # Return basic response if there's an error
            return {
                "filename": filename,
                "message": f"Error processing PDF: {str(e)}",
                "status": "error"
            }
    
    @staticmethod
    def format_response(filename: str, text: str, page_count: int, 
                        summary: str = None, knowledge_graph = None, 
                        document_id: str = None, db_result: Dict[str, Any] = None,
                        chunks_count: int = None) -> Dict[str, Any]:
        """
        Format the response for the API.
        
        Args:
            filename: Name of the processed PDF file
            text: Extracted text from the PDF
            page_count: Number of pages in the PDF
            summary: Summary of the PDF content
            knowledge_graph: Generated knowledge graph
            document_id: Document ID in the Neo4j database
            db_result: Result of database operation
            chunks_count: Number of chunks the document was split into
            
        Returns:
            Dictionary with the formatted response
        """
        response = {
            "filename": filename,
            "page_count": page_count,
            "message": "PDF processed successfully",
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
        if summary:
            response["summary"] = summary
            
        if knowledge_graph:
            response["knowledge_graph"] = {
                "nodes_count": len(knowledge_graph.nodes),
                "relationships_count": len(knowledge_graph.relationships)
            }
            
        if document_id:
            response["document_id"] = document_id
            
        if db_result:
            response["database_result"] = db_result
            
        if chunks_count is not None:
            response["chunks_processed"] = chunks_count
            
        return response
