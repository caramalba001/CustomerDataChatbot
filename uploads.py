from azure.cosmos import CosmosClient, PartitionKey
from transformers import AutoTokenizer, AutoModel
import json
import uuid
import torch

# Initialize the Hugging Face model and tokenizer
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize the Cosmos client
cosmos_url = "YOUR_COSMOS_URL"
cosmos_key = "YOUR_COSMOS_KEY"
database_name = "YOUR_DATABASE_NAME"
container_name = "YOUR_CONTAINER_NAME"

client_cosmos = CosmosClient(cosmos_url, cosmos_key)
database = client_cosmos.create_database_if_not_exists(id=database_name)
container = database.create_container_if_not_exists(
    id=container_name,
    partition_key=PartitionKey(path="/id"),  # Replace with your actual partition key path
)
print("Container created successfully in a serverless account.")

# File path to the JSON file
file_path = "YOUR_DATA_PATH_JSON"

# Read and parse JSON file
with open(file_path, "r", encoding="utf-8") as file:
    raw = json.load(file)

# Define chunk size
chunk_size = 50000

# Calculate number of chunks and split the data
num_chunks = len(raw) // chunk_size + (1 if len(raw) % chunk_size != 0 else 0)
data_chunks = [raw[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

# Print the number of chunks
print(f"Number of chunks: {num_chunks}")

data = data_chunks[0]

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().tolist()

def upload_batch_data(batch, container):
    """Prepare and upload batch data."""
    documents = []
    for record in batch:
        document = {
            "id": str(uuid.uuid4()),  # Generate a unique ID
            "name": record.get("name"),
            "customer_detail": record.get("customer_detail"),
            "product_buy": record.get("product_buy"),
            "embedding": record.get("embedding")
        }
        documents.append(document)

    # Upsert documents into the Cosmos DB container (replace with your database logic)
    for doc in documents:
        container.upsert_item(doc)

    # Upsert documents into the Cosmos DB container (replace with your database logic)
    for doc in documents:
        container.upsert_item(doc)

batch_size = 5

# Process data in batches
for i in range(0, len(data), batch_size):
    # Slice the data to create the current batch
    batch = data[i:i + batch_size]

    # Extract names from the batch
    names = [record.get("name", "").strip() for record in batch 
             if isinstance(record.get("name", ""), str) and record.get("name", "").strip()]

    # Debugging and validation
    print(f"Processing batch {i // batch_size + 1}/{(len(data) + batch_size - 1) // batch_size}")
    print(f"Number of valid names in batch: {len(names)}")
    
    # Ensure all items are valid strings
    assert all(isinstance(name, str) and name for name in names), "Not all items in `names` are valid strings"

    # Generate embeddings for the `name` field
    embeddings = [get_embedding(name, tokenizer, model) for name in names]
        
    # Attach embeddings back to the corresponding records
    for record, embedding in zip(batch, embeddings):
        record["embedding"] = embedding

    # Upload the batch data
    upload_batch_data(batch, container)
