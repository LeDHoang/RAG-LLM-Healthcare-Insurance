# RAG-LLM-Healthcare-Insurance

## Description

A Streamlit-based application for uploading PDF documents, splitting them into text chunks, generating embeddings using Amazon Bedrock Embeddings, creating a FAISS vector store locally and in S3. This enables retrieval-augmented generation (RAG) workflows for healthcare insurance documents.

## Features

- Upload and process PDF files via Streamlit UI.
- Split text into manageable chunks with overlap.
- Generate embeddings using AWS Bedrock Embeddings.
- Create and store FAISS vector indexes locally and in S3.
- Configurable via environment variables.

## Prerequisites

- Python 3.8 or higher
- AWS account with S3 and Bedrock access
- An existing S3 bucket for storing FAISS indexes

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd RAG-LLM-Healthcare-Insurance
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r Admin/requirements.txt
   ```
4. Create a `.env` file in the project root with the following variables:
   ```ini
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=your_region
   BUCKET_NAME=your_s3_bucket_name
   ```

## Running the Admin App

```bash
streamlit run Admin/admin.py --server.port 8501 --server.address 127.0.0.1
```

Open http://localhost:8501 in your browser, upload a PDF from `pdf-sources/`, and the app will process and upload FAISS indexes to S3.

## Testing

1. Use sample PDFs in `pdf-sources/`.
2. Verify generated index files in `/tmp` (e.g., `/tmp/<request_id>.bin.faiss` and `.pkl`).
3. Check your S3 bucket for `my_faiss.faiss` and `my_faiss.pkl`.
4. Troubleshoot via Streamlit console logs.

## Project Structure

```
Admin/
├─ admin.py
├─ requirements.txt
User/
└─ app.py (future user-facing interface)
pdf-sources/
└─ [Sample PDFs]
README.md
.gitignore
```