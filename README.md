# Sentiment Analyzer (Flask + NLTK / Transformers)

## Why this design
- **Meets assignment criteria**: Web UI, file upload, visualization (bar chart + label), NLP model, preprocessing, prediction.  
- **Robust & fast**: Uses **NLTK VADER** by default (plug-and-play, no GPU).  
- **Extensible**: Flip `USE_TRANSFORMER=1` to use **HuggingFace DistilBERT** for stronger semantics.  
- **Clear separation**: `nlp/preprocess.py` (tokenization, stopwords, lemmatization) and
  `nlp/sentiment.py` (model inference).  
- **Visualization**: Chart.js ensures an easy, legible bar chart.

## Running
See the top-level instructions in the main response. Open `http://127.0.0.1:5000`.

### Switch to transformer model (optional)
```bash
# inside your venv, after installing extra deps from requirements.txt
set USE_TRANSFORMER=1        # on Windows (cmd)
export USE_TRANSFORMER=1     # on macOS/Linux (bash/zsh)
python app.py
Challenges & notes
Preprocessing vs VADER: VADER benefits from keeping negations/emojis; heavy cleaning may slightly reduce performance. For pedagogy, we still show preprocessing and feed raw text to VADER while using the cleaned text for transformers.

Neutral class (transformer): SST-2 models are 2-class. We construct a neutral band from confidence around 0.5 to provide a 3-way output required by the UI.

Uploaded files: We support UTF-8 .txt. For other encodings, we fallback with errors="ignore".

Screenshots to capture for the report
Home page before input

Entered text & Analyze pressed

Output label (color) + bar chart

File upload flow with .txt

(Optional) Same text compared with VADER vs Transformer