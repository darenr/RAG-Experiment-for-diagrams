import openai
import hnswlib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
import locale
from loguru import logger
from joblib import Memory
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
import markdown

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
memory = Memory("./cachedir", verbose=0)


app = Flask(__name__, static_folder="static", static_url_path="/static")
client = openai.OpenAI()
index = None
documents: list[str] = None


def embed_document(text: str) -> list[float]:
    """
    Get the embedding for a text string. Return a list of 1536 floats
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


def create_index(docs: list[str]) -> tuple:
    """Create an index for the documents with hnswlib

    Args:
        docs (list[str]): List of document strings to index

    Returns:
        tuple: (index, documents) - The HNSW index, and original documents
    """
    global index, documents
    # Initialize the HNSW index
    dim = 1536  # Embedding dimension is 1536 floats
    num_elements = len(docs)

    documents = docs.copy()

    # Initialize the index with the specified dimension
    index = hnswlib.Index(space="cosine", dim=dim)
    logger.info("Initialized HNSW index")

    # Initialize the index - use default parameters for simplicity
    # M is the number of bidirectional links created for each new element during construction
    # ef_construction is the size of the dynamic list for the nearest neighbors
    index.init_index(max_elements=num_elements, ef_construction=100, M=16)

    embedding_fn = memory.cache(embed_document)

    # Create embeddings for all documents
    doc_embeddings = []
    for i, doc in enumerate(docs):
        embedding = embedding_fn(doc)
        doc_embeddings.append(embedding)

    logger.info(f"Created embeddings for {len(docs)} documents")

    # Add items to the index with sequential ids
    index.add_items(np.array(doc_embeddings), list(range(len(docs))))

    # Set ef (parameter controlling query time/accuracy trade-off) for search
    index.set_ef(200)  # Higher ef leads to better accuracy but slower search

    return index, docs


def get_context(query, k=3) -> str:
    """
    Get relevant context for a query using the HNSW index

    Args:
        query (str): The query text
        index: HNSW index object
        doc_embeddings: List of document embeddings
        docs: List of original document texts
        k (int): Number of relevant documents to retrieve

    Returns:
        str: Context string from the most relevant documents
    """
    global index, documents

    # Get embedding for the query
    query_embedding = embed_document(query)

    # Query the index for k nearest neighbors
    labels, distances = index.knn_query(query_embedding, k=5)

    # Collect the most relevant documents as context
    context_docs = [documents[idx] for idx in labels[0]]

    # Join them into a single context string
    return "\n\n- ".join(context_docs)


@app.route("/")
def home():
    """Serve the main search page"""
    return render_template("index.html")


@app.route("/static/page1.png")
def serve_diagram():
    """Serve the diagram image from the data directory"""
    return send_from_directory("data", "page1.png")


@app.route("/api/query", methods=["GET"])
def search():
    """Handle search requests"""
    query = request.args.get("query")
    logger.info(f"Search endpoint called with query: {query}")

    # Get context using the global index
    global index, documents
    context = get_context(query, documents)

    prompt = f"""
    You are a helpful assistant. Answer the user's question based only on the context provided.
    Context: {context}
    Question: {query}
    Answer:
    """

    print(Panel(Markdown(context), title="Context", subtitle="Relevant documents"))

    results = client.chat.completions.create(
        model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
    )
    # Convert the OpenAI response to a dictionary that can be properly jsonified
    result = markdown.markdown(results.choices[0].message.content)
    logger.info(f"Search results: {result}")

    return jsonify(result)


def load_documents(file_path="data/qa_pairs.jsonl"):
    """Load documents from a JSONL file"""
    import json

    docs = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                docs.append(f"Question: {data['question']}\nAnswer: {data['answer']}")

        logger.info(f"Loaded {len(docs)} documents from {file_path}")
        return docs
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


def main():
    # Load documents and create index
    docs = load_documents()
    if docs:
        create_index(docs)
    else:
        logger.warning("No documents loaded, running without document retrieval")

    # Start the Flask application
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
