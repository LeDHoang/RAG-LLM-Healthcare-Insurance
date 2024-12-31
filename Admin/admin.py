from flask import Flask, request, jsonify
import os
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.amazon_titan import AmazonTitanEmbeddings
import faiss
import boto3
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Amazon S3 Configuration
S3_BUCKET = 'your-s3-bucket-name'
s3_client = boto3.client('s3')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Save uploaded PDF
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Extract text from the PDF
    text = extract_text(file_path)
    
    # Split the text into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    # Create embeddings using Amazon Titan
    embedding_model = AmazonTitanEmbeddings()
    embeddings = [embedding_model.embed(chunk) for chunk in chunks]
    
    # Create FAISS index
    dimension = len(embeddings[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))
    
    # Save the FAISS index locally
    index_file = 'vector_index.faiss'
    faiss.write_index(faiss_index, index_file)
    
    # Upload the index to S3
    s3_client.upload_file(index_file, S3_BUCKET, index_file)
    
    return jsonify({'message': 'PDF processed and index uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True)
