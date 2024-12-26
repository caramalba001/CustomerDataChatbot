from flask import Flask, request, jsonify, render_template
from azure.cosmos import CosmosClient, PartitionKey
from transformers import AutoTokenizer, AutoModel
from openai import AzureOpenAI
import torch
import uuid
import json

app = Flask(__name__)

# Initialize the Cosmos client
cosmos_url = "YOUR_COSMOS_URL"
cosmos_key = "YOUR_COSMOS_KEY"
database_name = "YOUR_DATABASE_NAME"
container_name = "YOUR_CONTAINER_NAME"

client_cosmos = CosmosClient(cosmos_url, cosmos_key)
database = client_cosmos.create_database_if_not_exists(id=database_name)
container = database.create_container_if_not_exists(
    id=container_name,
    partition_key=PartitionKey(path="/id"),
)

# Initialize Azure OpenAI client
AI_API_KEY = "YOUR_AI_API_KEY"
AI_API_VERSION = "2024-02-01"
AI_ENDPOINT = "YOUR_AI_ENDPOINT"

client_AI = AzureOpenAI(api_key=AI_API_KEY, api_version=AI_API_VERSION, azure_endpoint=AI_ENDPOINT)

model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Global chat history
chat_history = []


def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().tolist()


def vector_search(input_text):
    query_embedding = get_embedding(input_text, tokenizer, model)
    results = container.query_items(
        query=f'''
        SELECT TOP 1 c.id, c.name, c.customer_detail, c.product_buy, VectorDistance(c.embedding, {query_embedding}) AS SimilarityScore  
        FROM c 
        ORDER BY VectorDistance(c.embedding, {query_embedding})
        ''',
        enable_cross_partition_query=True
    )
    return list(results)


def chat_AI(query):
    global chat_history

    # Retrieve customer information based on the query
    information = vector_search(query)

    # Build the chat history into the system message content
    history_content = "\n".join(
        [
            f"user: {item['user']}\nassistant: {item['assistant']}"
            for item in chat_history
        ]
    )

    # Call the Azure OpenAI model
    ai_response = client_AI.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""
                You are \"น้องวิริน,\" a friendly, joyful, and helpful customer support representative for วิริยะ Insurance. Your primary role is to assist customers by providing them with accurate information based on the company’s provided resources and guidelines. น้องวิริน is a cheerful young woman, always ready to listen carefully, address customer concerns with empathy, and answer questions in a polite, respectful manner that brings comfort and clarity to the customer.

                Please keep the following points in mind when responding:
                1. Polite Tone: Use polite language suitable for professional customer service in Thai, such as \"ค่ะ\" and \"ค่ะคุณลูกค้า\" to show respect and warmth. Your tone should make customers feel welcome, comfortable, and valued.
                2. Joyful and Friendly: Your tone should be light-hearted, cheerful, and friendly, like you’re smiling through your response. Make customers feel like they're talking to a helpful friend who truly cares about their needs.
                3. Clear and Accurate Information: Always provide clear, straightforward information. Stick to the details in the source provided. If the customer asks about topics outside the given information, kindly redirect them to contact a human representative for further help.
                4. Empathy in Response: If a customer has a concern, such as a complaint or claim issue, respond with empathy. Acknowledge their concern, show understanding, and provide helpful next steps or clarification with a calm and supportive tone.

                Here is the customer information:
                {information}

                And here is the chat history:
                {history_content}

                Use chat history to understand the ongoing conversation and respond naturally. Supplement your response with customer information only when relevant or when asked for specific details.

                """
            },
            {"role": "user", "content": query}
        ]
    )

    ai_output = ai_response.choices[0].message.content

    # Record the query and response in the chat history
    chat_history.append({"user": query, "assistant": ai_output})

    return ai_output


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        ai_response = chat_AI(user_input)
        return jsonify({"reply": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared successfully"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)