from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from services.pdf_service import PDFService
from services.embedding_service import EmbeddingService
from typing import Optional

# Create router for PDF-related endpoints
router = APIRouter(
    prefix="/pdf",
    tags=["pdf"],
    responses={404: {"description": "Not found"}},
)

@router.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and extract its text content.
    
    Args:
        file: The uploaded PDF file
        
    Returns:
        JSONResponse with the filename, page count, and text preview
        
    Raises:
        HTTPException: If the file is not a PDF or if there's an error processing it
    """
    # Check if the uploaded file is a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Use the PDF service to extract text
        text, page_count = PDFService.extract_text_from_pdf(contents)
        
        # Print the extracted text to console
        print(f"Extracted text from {file.filename}:")
        print(text)
        
        # Format the response using the service
        response_data = PDFService.format_response(file.filename, text, page_count)
        
        # Process PDF with knowledge graph
        kg_response_data = await PDFService.process_pdf_with_knowledge_graph(contents, file.filename)
        
        return JSONResponse(content={**response_data, **kg_response_data})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Reset file pointer and close
        await file.seek(0)
        await file.close()

@router.post("/process-with-kg/")
async def process_pdf_with_knowledge_graph(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, generate a knowledge graph, and store it in Neo4j.
    
    Args:
        file: The uploaded PDF file
        
    Returns:
        JSONResponse with the processing results including knowledge graph information
        
    Raises:
        HTTPException: If the file is not a PDF or if there's an error processing it
    """
    # Check if the uploaded file is a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Process the PDF with knowledge graph generation and Neo4j storage
        response_data = await PDFService.process_pdf_with_knowledge_graph(contents, file.filename)
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF with knowledge graph: {str(e)}")
    finally:
        # Reset file pointer and close
        await file.seek(0)
        await file.close()

@router.get("/graph/{document_id}")
async def get_knowledge_graph(document_id: str):
    """
    Retrieve a knowledge graph for a specific document from Neo4j.
    
    Args:
        document_id: Unique identifier for the document
        
    Returns:
        JSONResponse with the knowledge graph data
        
    Raises:
        HTTPException: If there's an error retrieving the knowledge graph
    """
    try:
        from services.neo4j_service import Neo4jService
        
        # Initialize Neo4j service
        neo4j_service = Neo4jService()
        
        # Get knowledge graph from Neo4j
        graph_data = neo4j_service.get_document_graph(document_id)
        neo4j_service.close()
        
        if graph_data.get("status") == "error":
            raise HTTPException(status_code=500, detail=f"Error retrieving knowledge graph: {graph_data.get('error')}")
        
        return JSONResponse(content=graph_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving knowledge graph: {str(e)}")

@router.get("/search/{document_id}")
async def similarity_search(
    document_id: str, 
    query: str, 
    top_k: Optional[int] = Query(5, description="Number of top results to return")
):
    """
    Perform a similarity search on nodes in a document using embeddings.
    
    Args:
        document_id: Unique identifier for the document to search within
        query: The search query text
        top_k: The number of top results to return (default: 5)
        
    Returns:
        JSONResponse with the search results including similarity scores
        
    Raises:
        HTTPException: If there's an error performing the search
    """
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Perform similarity search
        search_results = embedding_service.similarity_search(query, document_id, top_k)
        
        if not search_results:
            return JSONResponse(content={
                "document_id": document_id,
                "query": query,
                "results": [],
                "message": "No results found or error occurred during search"
            })
        
        return JSONResponse(content={
            "document_id": document_id,
            "query": query,
            "results": search_results,
            "count": len(search_results)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {str(e)}")

@router.post("/update-embeddings/{document_id}")
async def update_node_embeddings(document_id: str):
    """
    Update embeddings for all nodes in a document.
    
    Args:
        document_id: Unique identifier for the document to update embeddings for
        
    Returns:
        JSONResponse with information about the update
        
    Raises:
        HTTPException: If there's an error updating the embeddings
    """
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Update embeddings
        result = await embedding_service.update_node_embeddings(document_id)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=f"Error updating embeddings: {result.get('error')}")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating embeddings: {str(e)}")
