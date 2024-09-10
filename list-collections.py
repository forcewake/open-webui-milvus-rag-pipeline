from pymilvus import connections, list_collections

# Connect to the Milvus server
connections.connect("default", host="localhost", port="19530")  # Adjust host and port as necessary

# List all collections
collections = list_collections()

print("Collections in Milvus:", collections)

