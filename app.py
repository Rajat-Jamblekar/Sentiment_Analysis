import os
from flask import Flask, render_template, request, jsonify
from nlp.preprocess import preprocess_text
from nlp.sentiment import SentimentAnalyzer

# Toggle model: 0 = VADER (default), 1 = Transformers
USE_TRANSFORMER = int(os.environ.get("USE_TRANSFORMER", "1"))

app = Flask(__name__, static_folder="static", template_folder="templates")
analyzer = SentimentAnalyzer(use_transformer=USE_TRANSFORMER)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_name=analyzer.model_name)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accepts either raw text or an uploaded .txt file.
    Returns: {label, scores: {positive, neutral, negative}, detail}
    """
    text = request.form.get("text", "").strip()
    uploaded = request.files.get("file")

    contents = text
    if uploaded and uploaded.filename:
        try:
            file_bytes = uploaded.read()
            contents = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return jsonify({"error": "Unable to read file. Use UTF-8 .txt only."}), 400

    if not contents:
        return jsonify({"error": "No input provided."}), 400

    # Preprocess (tokenization, stopword removal, lemmatization, etc.)
    preprocessed = preprocess_text(contents)

    # For academic clarity: run sentiment on original text
    # (VADER often performs best on near-raw text with emojis/negations)
    result = analyzer.predict(contents, preprocessed_text=preprocessed)

    return jsonify(result), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)