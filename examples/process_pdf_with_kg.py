"""
Example script to demonstrate processing a PDF file with knowledge graph generation and Neo4j integration.
"""
import os
import sys
import logging
import requests
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def process_pdf_with_kg(pdf_path, api_url="http://localhost:8000"):
    """
    Process a PDF file with knowledge graph generation and Neo4j integration.
    
    Args:
        pdf_path: Path to the PDF file
        api_url: URL of the API server
        
    Returns:
        Response from the API
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            logging.error(f"File not found: {pdf_path}")
            return None
        
        # Check if file is a PDF
        if not pdf_path.lower().endswith('.pdf'):
            logging.error(f"File is not a PDF: {pdf_path}")
            return None
        
        # Get filename
        filename = os.path.basename(pdf_path)
        
        # Open file
        with open(pdf_path, 'rb') as f:
            # Create files dictionary
            files = {'file': (filename, f, 'application/pdf')}
            
            # Send request to API
            logging.info(f"Sending request to {api_url}/pdf/process-with-kg/")
            response = requests.post(f"{api_url}/pdf/process-with-kg/", files=files)
            
            # Check response
            if response.status_code == 200:
                logging.info("PDF processed successfully")
                return response.json()
            else:
                logging.error(f"Error processing PDF: {response.text}")
                return None
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return None

def retrieve_knowledge_graph(document_id, api_url="http://localhost:8000"):
    """
    Retrieve a knowledge graph for a specific document from Neo4j.
    
    Args:
        document_id: Unique identifier for the document
        api_url: URL of the API server
        
    Returns:
        Response from the API
    """
    try:
        # Send request to API
        logging.info(f"Sending request to {api_url}/pdf/graph/{document_id}")
        response = requests.get(f"{api_url}/pdf/graph/{document_id}")
        
        # Check response
        if response.status_code == 200:
            logging.info("Knowledge graph retrieved successfully")
            return response.json()
        else:
            logging.error(f"Error retrieving knowledge graph: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error retrieving knowledge graph: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a PDF file with knowledge graph generation and Neo4j integration")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of the API server")
    parser.add_argument("--retrieve-only", help="Only retrieve knowledge graph for the specified document ID")
    
    args = parser.parse_args()
    
    if args.retrieve_only:
        # Retrieve knowledge graph
        result = retrieve_knowledge_graph(args.retrieve_only, args.api_url)
        if result:
            print("Knowledge Graph:")
            print(f"Document ID: {result['document_id']}")
            print(f"Nodes: {len(result['nodes'])}")
            print(f"Relationships: {len(result['relationships'])}")
            
            # Print nodes
            print("\nNodes:")
            for i, node in enumerate(result['nodes']):
                print(f"{i+1}. {node['id']}: {node['description'][:50]}...")
            
            # Print relationships
            print("\nRelationships:")
            for i, rel in enumerate(result['relationships']):
                print(f"{i+1}. {rel['source']} --[{rel['type']}]--> {rel['target']}")
    else:
        # Process PDF
        result = process_pdf_with_kg(args.pdf_path, args.api_url)
        if result:
            print("PDF Processed Successfully:")
            print(f"Filename: {result['filename']}")
            print(f"Page Count: {result['page_count']}")
            print(f"Document ID: {result['document_id']}")
            
            # Print knowledge graph info
            if 'knowledge_graph' in result:
                print(f"Knowledge Graph Nodes: {result['knowledge_graph']['nodes_count']}")
                print(f"Knowledge Graph Relationships: {result['knowledge_graph']['relationships_count']}")
            
            # Print database result
            if 'database_result' in result:
                print(f"Database Status: {result['database_result']['status']}")
                print(f"Nodes Committed: {result['database_result'].get('nodes_committed', 0)}")
                print(f"Relationships Committed: {result['database_result'].get('relationships_committed', 0)}")
            
            # Show how to retrieve the knowledge graph
            print("\nTo retrieve this knowledge graph later, run:")
            print(f"python {os.path.basename(__file__)} --retrieve-only {result['document_id']} --api-url {args.api_url}")
