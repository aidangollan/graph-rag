# graph-rag

## PDF Processing API

A FastAPI-based API for processing PDF documents.

### Features

- Upload PDF files
- Extract raw text from PDF documents
- Simple JSON response with text preview

### Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API:
   ```
   uvicorn main:app --reload
   ```

3. Access the API:
   - API documentation: http://localhost:8000/docs
   - Upload endpoint: http://localhost:8000/upload-pdf/

### API Endpoints

- `POST /upload-pdf/`: Upload a PDF file and extract text
- `GET /`: Basic API information