"""
Main application entry point.
"""
import logging
import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes.pdf_routes import router as pdf_router
from routes.chat_routes import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="PDF Processing API with Knowledge Graph",
    description="API for processing PDF documents with knowledge graphs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pdf_router)
app.include_router(chat_router)

@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        API information
    """
    return {
        "message": "PDF Processing API with Knowledge Graph",
        "endpoints": {
            "PDF Upload": "/pdf/upload/",
            "Process with KG": "/pdf/process-with-kg/",
            "Get Graph": "/pdf/graph/{document_id}",
            "Similarity Search": "/pdf/search/{document_id}?query={query}&top_k={top_k}",
            "Update Embeddings": "/pdf/update-embeddings/{document_id}",
            "Chat": "/chat/{document_id}?query={query}&top_k={top_k}",
            "Get All Documents": "/pdf/documents"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler.
    
    Args:
        request: The request that caused the exception
        exc: The exception
        
    Returns:
        JSON response with error details
    """
    logging.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
