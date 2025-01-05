
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
from langchain_community.vectorstores import FAISS
# import numpy as np

#Amazon Titan Configuration
# from langchain_community.embeddings import AmazonTitanEmbeddings

#pdf loader
from langchain_community.document_loaders import PyPDFLoader

bedrock_client = boto3.client("service_name=bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)
def split_text(text,chunk_size,chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(text)
    return chunks
#create vector store
def create_vector_store(request_id,chunks):
    #using FAISS
    vector_store_faiss = FAISS.from_documents(chunks,BedrockEmbeddings())
    #Create embeddings using Bedrock
    embeddings = BedrockEmbeddings()
    #Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.dimension)
    index.add(np.array(embeddings))
    return index

def main():
    st.title("PDF Upload and Processing")
    st.write("Upload a PDF file to process it.")
    st.write("This is the admin page for the PDF upload and processing system.")
    #Upload the file
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        #get the request id
        request_id = str(uuid.uuid4())
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
        #Split the text into chunks using the split_text function
        chunks = split_text(pages,1000,200)
        st.write(f"Splitting the text into chunks. Splitted Documents length: {len(chunks)}")
        st.write("--------------------------------")
        st.write(chunks[0])
        st.write("--------------------------------")
        st.write(chunks[1])
        st.write("--------------------------------")
        st.write(chunks[2])
        st.write("--------------------------------")

        st.write("Creating vector store")
        result = create_vector_store(request_id,chunks)
        st.write(result)
        if result:
            st.write("Vector store created successfully")
        else:
            st.write("Vector store creation failed")
if __name__ == '__main__':
    main()