import openai
from flask import Flask, jsonify, render_template, request, send_from_directory
import locale
from loguru import logger

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

app = Flask(__name__, static_folder="static", static_url_path="/static")

client = openai.OpenAI()


def get_context(query) -> str:
    return "key takeaway is 50% of GenAI budgets go to sales and marketing."


@app.route("/")
def index():
    """Serve the main search page"""
    return render_template("index.html")


@app.route("/static/page1.png")
def serve_diagram():
    """Serve the diagram image from the data directory"""
    return send_from_directory("data", "page1.png")


@app.route("/api/query", methods=["GET"])
def search():
    """Handle search requests"""
    logger.info("Search endpoint called")
    query = request.args.get("query")
    prompt = f"""
    You are a helpful assistant. Answer the user's question based on the context provided.
    Context: {get_context(query)}
    Question: {query}
    Answer:
    """

    results = client.chat.completions.create(
        model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
    )
    # Convert the OpenAI response to a dictionary that can be properly jsonified
    return jsonify(results.model_dump())


def main():
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
