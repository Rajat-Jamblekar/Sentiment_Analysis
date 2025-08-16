import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE
)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def basic_clean(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = USER_RE.sub(" ", text)     # remove @mentions
    text = HASHTAG_RE.sub(r"\1", text) # keep hashtag words
    text = EMOJI_RE.sub(" ", text)
    return text

def preprocess_text(text: str) -> str:
    """
    1) Remove URLs, mentions, emojis
    2) Lowercase
    3) Tokenize
    4) Remove punctuation & stopwords
    5) Lemmatize
    Returns a whitespace-joined, cleaned string.
    """
    text = basic_clean(text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    clean_tokens = []
    for tok in tokens:
        if tok in string.punctuation:
            continue
        if tok in stop_words:
            continue
        lemma = lemmatizer.lemmatize(tok)
        clean_tokens.append(lemma)

    return normalize_whitespace(" ".join(clean_tokens))