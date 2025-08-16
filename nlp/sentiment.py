from typing import Dict, Any
import math

# VADER (rule/lexicon-based)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Optional: Transformers (DistilBERT fine-tuned for sentiment)
try:
    from transformers import pipeline
except Exception:
    pipeline = None  # handled at runtime

class SentimentAnalyzer:
    def __init__(self, use_transformer: int = 0):
        self.use_transformer = bool(use_transformer)
        self.vader = SentimentIntensityAnalyzer()

        self.hf = None
        if self.use_transformer:
            if pipeline is None:
                raise RuntimeError(
                    "transformers not available; install extras from requirements.txt."
                )
            # Load the sentiment analysis pipeline for Transformers
            self.hf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.model_name = "Transformers (DistilBERT)" if self.use_transformer else "NLTK VADER"

    def _vader_predict(self, text: str) -> Dict[str, Any]:
        scores = self.vader.polarity_scores(text)
        # scores: {neg, neu, pos, compound}
        neg = float(scores["neg"])
        neu = float(scores["neu"])
        pos = float(scores["pos"])

        # Label via compound thresholding (common practice)
        comp = scores["compound"]
        if comp >= 0.05:
            label = "positive"
        elif comp <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "scores": {"positive": pos, "neutral": neu, "negative": neg},
            "detail": {"compound": comp}
        }

    def _hf_predict(self, text: str) -> Dict[str, Any]:
        out = self.hf(text, truncation=True)[0]
        raw_label = out["label"].lower()  # 'positive' or 'negative'
        conf = float(out["score"])

        # Create a 3-way distribution for display
        # Heuristic: if confidence near 0.5, treat as neutralish.
        # Convert to pos/neg with a small neutral band.
        pos_prob = conf if raw_label == "positive" else 1.0 - conf
        neg_prob = 1.0 - pos_prob
        # Neutral band: closer to 0.5 → higher neutral
        neutral = max(0.0, 1.0 - 2.0 * abs(pos_prob - 0.5))
        # Renormalize to sum to 1
        s = pos_prob + neg_prob + neutral
        pos_prob, neg_prob, neutral = pos_prob/s, neg_prob/s, neutral/s

        if pos_prob > 0.5 and pos_prob - neg_prob > 0.1:
            label = "positive"
        elif neg_prob > 0.5 and neg_prob - pos_prob > 0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "scores": {
                "positive": round(pos_prob, 4),
                "neutral": round(neutral, 4),
                "negative": round(neg_prob, 4),
            },
            "detail": {
                "raw_label": raw_label,
                "confidence": round(conf, 4)
            }
        }

    def predict(self, text: str, preprocessed_text: str = "") -> Dict[str, Any]:
        """
        Returns {label, scores: {positive, neutral, negative}, detail}
        """
        if self.use_transformer:
            return self._hf_predict(preprocessed_text or text)
        return self._vader_predict(text)
