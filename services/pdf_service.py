import PyPDF2
import io
import uuid
import logging
from typing import Tuple, Dict, Any

from services.kg_service import CustomKGBuilder
from services.neo4j_service import Neo4jService
from utils.llm import get_llm
from utils.constants import GPT_4O_MINI
from langchain_core.prompts import ChatPromptTemplate

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
    def generate_summary(text: str) -> str:
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
            summary = chain.invoke({"text": text})
            
            return summary.content
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            # Return a short excerpt of the text if summarization fails
            return text[:500] + "..." if len(text) > 500 else text
    
    @staticmethod
    def process_pdf_with_knowledge_graph(pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a PDF document, extract text, generate knowledge graph, and store in Neo4j.
        
        Args:
            pdf_content: Raw bytes of the PDF file
            filename: Name of the PDF file
            
        Returns:
            Dictionary with the processing results
        """
        try:
            # Extract text from PDF
            text, page_count = PDFService.extract_text_from_pdf(pdf_content)
            
            # Generate summary
            summary = PDFService.generate_summary(text)
            
            # Generate document ID
            document_id = f"{filename.replace('.pdf', '')}_{uuid.uuid4().hex[:8]}"
            
            # Generate knowledge graph
            knowledge_graph = CustomKGBuilder.create_knowledge_graph(text, summary)
            
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
                db_result=db_result
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
                        document_id: str = None, db_result: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
        return response
