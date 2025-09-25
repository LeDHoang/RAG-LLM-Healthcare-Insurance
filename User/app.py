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

def _build_context_sections(docs):
    """Format retrieved docs into numbered sections with source hints for citations."""
    sections = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
        page = doc.metadata.get('page', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
        content = (doc.page_content or '').strip()
        sections.append(
            f"[S{idx}]\n{content}\n(Source: {source}, page {page})"
        )
    return "\n\n".join(sections)

def _build_converse_messages(question, docs):
    """Create safety-guided user message for Nova Converse (no system role supported)."""
    context_text = _build_context_sections(docs)

    system_instructions = (
        "You are a helpful assistant for healthcare insurance documents. "
        "Follow these rules strictly: \n"
        "- Ground every answer ONLY in the provided Context sections.\n"
        "- If the answer is not in context, say you don't know and suggest checking the policy documents.\n"
        "- Include inline citations using the section IDs like [S1], [S2] wherever specific facts are used.\n"
        "- Be concise, neutral, and precise. Avoid speculation or fabrication.\n"
        "- Do not provide legal, medical, or financial advice. Provide informational guidance only.\n"
        "- Do not output secrets, credentials, or personal data.\n"
        "- If the user asks for actions that could cause harm or are outside scope, refuse briefly and provide safer alternatives.\n"
        "- Prefer bullet lists for multi-part answers.\n"
    )

    user_prompt = (
        f"{system_instructions}\n\n"
        f"Context sections (use for grounding and citations):\n\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations like [S1], [S2]. If unknown, say you don't know."
    )

    return [
        {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
    ]

def query_documents(question, vector_store, bedrock_client):
    """Query the documents using RAG"""
    try:
        # Search for relevant documents
        docs = vector_store.similarity_search(question, k=3)
        
        if not docs:
            return "No relevant documents found."
        
        # Build safety-guided messages for Nova Lite
        messages = _build_converse_messages(question, docs)

        # Use Converse API for Nova Lite
        try:
            response = bedrock_client.converse(
                modelId="us.amazon.nova-lite-v1:0",
                messages=messages,
                inferenceConfig={
                    "maxTokens": 512,
                    "temperature": 0.2
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
