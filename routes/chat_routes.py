"""
Routes for chat functionality with knowledge graph RAG.
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
from services.chat_service import ChatService

# Create router for chat-related endpoints
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

@router.post("/{document_id}")
async def chat_with_document(
    document_id: str,
    query: str,
    top_k: Optional[int] = Query(5, description="Number of top results to return")
):
    """
    Chat with a document using the knowledge graph.
    
    Args:
        document_id: Unique identifier for the document to chat with
        query: The user's query
        top_k: The number of top results to return (default: 5)
        
    Returns:
        JSONResponse with the chat response, used context nodes, and used context chunks
        
    Raises:
        HTTPException: If there's an error processing the chat query
    """
    try:
        # Initialize chat service
        chat_service = ChatService()
        
        # Process chat query
        response_data = await chat_service.process_chat_query(document_id, query)
        
        if "error" in response_data:
            raise HTTPException(status_code=500, detail=f"Error processing chat query: {response_data['error']}")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}")
