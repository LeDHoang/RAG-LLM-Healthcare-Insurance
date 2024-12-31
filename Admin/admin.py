
import boto3
import streamlit as st
import os
import uuid 
# Amazon S3 Configuration
S3_BUCKET = os.getenv('BUCKET_NAME')
s3_client = boto3.client('s3')

#Bedrock Configuration
from langchain_community.embeddings import BedrockEmbeddings

#Text Splitter | Split into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

#FAISS Configuration
import faiss
# import numpy as np

#Amazon Titan Configuration
# from langchain_community.embeddings import AmazonTitanEmbeddings

#pdf loader
from langchain_community.document_loaders import PyPDFLoader

# #Extract text from pdf
# def extract_text(file_path):
#     loader = PyPDFLoader(file_path)
#     documents = loader.load()
#     return documents

# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     file = request.files.get('file')
#     if not file:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     # Save uploaded PDF
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)
    
#     # Extract text from the PDF
#     text = extract_text(file_path)
    
#     # Split the text into chunks
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_text(text)
    
#     # Create embeddings using Amazon Titan
#     embedding_model = AmazonTitanEmbeddings()
#     embeddings = [embedding_model.embed(chunk) for chunk in chunks]
    
#     # Create FAISS index
#     dimension = len(embeddings[0])
#     faiss_index = faiss.IndexFlatL2(dimension)
#     faiss_index.add(np.array(embeddings))
    
#     # Save the FAISS index locally
#     index_file = 'vector_index.faiss'
#     faiss.write_index(faiss_index, index_file)
    
#     # Upload the index to S3
#     s3_client.upload_file(index_file, S3_BUCKET, index_file)
    
#     return jsonify({'message': 'PDF processed and index uploaded successfully'})

def main():
    st.title("PDF Upload and Processing")
    st.write("Upload a PDF file to process it.")
    st.write("This is the admin page for the PDF upload and processing system.")
    #Upload the file
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        request_id = uuid()
        st.write(f"Request ID: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        #Save the uploaded file to the local directory  
        with open(saved_file_name, "wb") as w:
            #getvalue() returns the file content as bytes
            w.write(uploaded_file.getvalue())
        #Load the file into a Langchain loader
        loader = PyPDFLoader(saved_file_name)
        #Split the file into pages
        pages = loader.load_and_split()
        st.write(f"Loaded {len(pages)} pages")

        # file_path = os.path.join(UPLOAD_FOLDER, saved_file_name)
        # uploaded_file.save(file_path)
        # st.success("File uploaded successfully")

if __name__ == '__main__':
    main()