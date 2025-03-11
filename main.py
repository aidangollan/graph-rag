from fastapi import FastAPI
from routes.pdf_routes import router as pdf_router
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(title="PDF Processing API with Knowledge Graph")

# Include routers
app.include_router(pdf_router)

@app.get("/")
async def root():
    """
    Root endpoint that provides basic API information.
    """
    return {
        "message": "PDF Processing API with Knowledge Graph", 
        "endpoints": [
            "/pdf/upload/",
            "/pdf/process-with-kg/",
            "/pdf/graph/{document_id}"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
