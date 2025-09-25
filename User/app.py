import streamlit as st
import boto3
import os
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock

# Load environment variables
load_dotenv()

# AWS Configuration
S3_BUCKET = os.getenv('BUCKET_NAME')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

def initialize_bedrock():
    """Initialize Bedrock client and embeddings"""
    try:
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION
        )
        
        bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", 
            client=bedrock_client
        )
        
        # Use cross-region inference profile for Nova Lite
        # Note: Nova Lite requires the Converse API, so we'll handle it manually in query_documents
        bedrock_llm = bedrock_client  # We'll use the client directly for Nova Lite
        
        return bedrock_embeddings, bedrock_llm
    except Exception as e:
        st.error(f"Failed to initialize Bedrock: {str(e)}")
        return None, None

def load_vector_store():
    """Load FAISS vector store from S3"""
    try:
        s3_client = boto3.client('s3')
        
        # Download FAISS files from S3
        s3_client.download_file(S3_BUCKET, "my_faiss.faiss", "/tmp/my_faiss.faiss")
        s3_client.download_file(S3_BUCKET, "my_faiss.pkl", "/tmp/my_faiss.pkl")
        
        # Load embeddings
        bedrock_embeddings, _ = initialize_bedrock()
        if bedrock_embeddings is None:
            return None
            
        # Load vector store
        vector_store = FAISS.load_local(
            folder_path="/tmp",
            index_name="my_faiss",
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        
        return vector_store
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

def query_documents(question, vector_store, bedrock_client):
    """Query the documents using RAG"""
    try:
        # Search for relevant documents
        docs = vector_store.similarity_search(question, k=3)
        
        if not docs:
            return "No relevant documents found."
        
        # Combine document content
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt for Nova Lite
        prompt = f"""Based on the following healthcare insurance documents, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Use Converse API for Nova Lite
        try:
            response = bedrock_client.converse(
                modelId="us.amazon.nova-lite-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                inferenceConfig={
                    "maxTokens": 512,
                    "temperature": 0.7
                }
            )
            
            # Extract the response text
            if 'output' in response and 'message' in response['output']:
                return response['output']['message']['content'][0]['text']
            else:
                return "Unable to generate response from Nova Lite."
                
        except AttributeError as e:
            return f"Error: Converse API not available. Please restart the application. Details: {str(e)}"
        except Exception as e:
            # Fallback error message
            return f"Error with Nova Lite generation: {str(e)}"
            
    except Exception as e:
        return f"Error querying documents: {str(e)}"

def main():
    st.title("Healthcare Insurance RAG Assistant")
    st.write("Ask questions about healthcare insurance documents")
    
    # Check if environment is configured
    if not all([os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY'), os.getenv('BUCKET_NAME')]):
        st.error("Please configure your AWS credentials and bucket name in the .env file")
        st.stop()
    
    # Initialize components
    with st.spinner("Initializing AI components..."):
        bedrock_embeddings, bedrock_llm = initialize_bedrock()
        
        if bedrock_embeddings is None or bedrock_llm is None:
            st.error("Failed to initialize Bedrock components. Please check your AWS configuration.")
            st.stop()
    
    # Load vector store
    with st.spinner("Loading document index..."):
        vector_store = load_vector_store()
        
        if vector_store is None:
            st.error("Failed to load document index. Please ensure documents have been processed via the Admin interface.")
            st.stop()
    
    st.success("✅ System ready! You can now ask questions about healthcare insurance documents.")
    
    # Question input
    question = st.text_input("Enter your question about healthcare insurance:")
    
    if st.button("Ask Question") and question:
        with st.spinner("Searching documents and generating answer..."):
            answer = query_documents(question, vector_store, bedrock_llm)
            
            st.subheader("Answer:")
            st.write(answer)
    
    # Example questions
    with st.expander("Example Questions"):
        st.write("""
        - What is a deductible?
        - What services are covered under preventive care?
        - How does copayment work?
        - What is the difference between in-network and out-of-network providers?
        - What are essential health benefits?
        """)

if __name__ == '__main__':
    main()
