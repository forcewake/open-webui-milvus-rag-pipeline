import os
import sys
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from docx import Document
import re

# Connect to Milvus
def connect_to_milvus(host='localhost', port='19530'):
    connections.connect(host=host, port=port)
    print("Connected to Milvus")

# Create a collection in Milvus
def create_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="Document collection")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create an IVF_FLAT index for fast retrieval
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created collection: {collection_name}")
    return collection

# Generate embeddings for documents
def generate_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents)
    return embeddings.tolist()

# Upload documents to Milvus
def upload_to_milvus(collection, documents, embeddings):
    entities = [
        embeddings,  # List of embeddings
        documents,   # List of documents
    ]
    insert_result = collection.insert(entities)
    print(f"Inserted {insert_result.insert_count} entities")

# Read and process the .docx file
def process_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():  # Only include non-empty paragraphs
            full_text.append(para.text.strip())
    return full_text

# Read and process .txt files
def process_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().strip()
    return text.splitlines()

# Split text into chunks
def split_into_chunks(text, max_chunk_size=500):
    chunks = []
    current_chunk = ""
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Process and upload files in a directory
def process_directory(directory_path, collection_name):
    # Connect to Milvus
    connect_to_milvus()

    # Create collection
    collection = create_collection(collection_name, dim=384)  # 384 is the dimension for 'all-MiniLM-L6-v2' model

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".docx"):
                print(f"Processing DOCX file: {file_path}")
                paragraphs = process_docx(file_path)
            elif file.endswith(".txt"):
                print(f"Processing TXT file: {file_path}")
                paragraphs = process_txt(file_path)
            else:
                print(f"Skipping unsupported file format: {file}")
                continue

            # Split paragraphs into chunks
            chunks = []
            for paragraph in paragraphs:
                chunks.extend(split_into_chunks(paragraph))

            # Generate embeddings
            embeddings = generate_embeddings(chunks)

            # Upload to Milvus
            upload_to_milvus(collection, chunks, embeddings)
            print(f"File {file_path} processed and uploaded successfully")

# Main function to accept command line arguments
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python milvus_uploader.py <directory_path> <collection_name>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    collection_name = sys.argv[2]
    
    process_directory(directory_path, collection_name)
