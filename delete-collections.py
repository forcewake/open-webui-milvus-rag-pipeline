from pymilvus import connections, Collection

# Connect to the Milvus server
connections.connect("default", host="localhost", port="19530")  # Adjust host and port as necessary

# Specify the collection you want to delete
collection_name = "faqs"

# Load the collection object
collection = Collection(name=collection_name)

# Drop the collection
collection.drop()

print(f"Collection '{collection_name}' has been successfully deleted.")

