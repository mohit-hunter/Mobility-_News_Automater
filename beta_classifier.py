"""
Beta Classifier Service for Frost News Collector

Loads the baseline classifier model and provides classification
for news items using precomputed embeddings.
"""

import json
import pickle
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import numpy as np

from config import (
    CLASSIFIER_MODEL_PATH, 
    EMBEDDING_PATH, 
    EMBEDDING_INDEX_PATH,
    BETA_CORRECTIONS_PATH,
    CATEGORIES
)


# Load model and embeddings at module import time
_classifier = None
_label_encoder = None
_embeddings = None
_embedding_index = None


def _load_model():
    """Load the classifier model and label encoder."""
    global _classifier, _label_encoder
    if _classifier is not None:
        return
    
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        print(f"[WARN] Classifier model not found at {CLASSIFIER_MODEL_PATH}")
        return
    
    try:
        with open(CLASSIFIER_MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(model_data, dict):
            _classifier = model_data.get('classifier') or model_data.get('model')
            _label_encoder = model_data.get('label_encoder')
        else:
            _classifier = model_data
            _label_encoder = None
        
        print(f"[OK] Loaded classifier model from {CLASSIFIER_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load classifier: {e}")


def _load_embeddings():
    """Load precomputed embeddings and index."""
    global _embeddings, _embedding_index
    if _embeddings is not None:
        return
    
    if os.path.exists(EMBEDDING_PATH):
        try:
            _embeddings = np.load(EMBEDDING_PATH)
            print(f"[OK] Loaded embeddings: {_embeddings.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load embeddings: {e}")
    
    if os.path.exists(EMBEDDING_INDEX_PATH):
        try:
            with open(EMBEDDING_INDEX_PATH, 'r', encoding='utf-8') as f:
                _embedding_index = json.load(f)
            print(f"[OK] Loaded embedding index: {len(_embedding_index)} entries")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding index: {e}")


def _init():
    """Initialize model and embeddings."""
    _load_model()
    _load_embeddings()


# Initialize on import
_init()


def generate_news_id(source_url: str) -> str:
    """Generate a stable news_id from the source URL using SHA256."""
    return hashlib.sha256(source_url.encode('utf-8')).hexdigest()[:16]


def _load_corrections() -> Dict[str, dict]:
    """Load corrections and return as dict keyed by news_id."""
    if not os.path.exists(BETA_CORRECTIONS_PATH):
        return {}
    
    try:
        with open(BETA_CORRECTIONS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        corrections = {}
        for c in data.get('corrections', []):
            news_id = c.get('news_id')
            if news_id:
                # Keep latest correction for each news_id
                corrections[news_id] = c
        return corrections
    except Exception:
        return {}


def _save_correction(correction: dict) -> None:
    """Append a correction to the corrections file."""
    if os.path.exists(BETA_CORRECTIONS_PATH):
        with open(BETA_CORRECTIONS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {'corrections': []}
    
    data['corrections'].append(correction)
    
    with open(BETA_CORRECTIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_correction(news_id: str) -> Optional[str]:
    """
    Get the corrected label for a news item if one exists.
    Returns the corrected_label or None.
    """
    corrections = _load_corrections()
    correction = corrections.get(news_id)
    if correction:
        return correction.get('corrected_label')
    return None


def save_correction(
    news_id: str, 
    original_prediction: str, 
    corrected_label: str, 
    reviewer_id: str
) -> dict:
    """
    Save a human label correction.
    Returns the correction record.
    """
    if corrected_label not in CATEGORIES:
        raise ValueError(f"Invalid category: {corrected_label}. Must be one of {CATEGORIES}")
    
    correction = {
        'news_id': news_id,
        'original_prediction': original_prediction,
        'corrected_label': corrected_label,
        'reviewer_id': reviewer_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    _save_correction(correction)
    return correction


def classify_single(news_id: str, source_url: str) -> Tuple[str, float]:
    """
    Classify a single news item.
    Returns (category, confidence).
    
    Falls back to 'government' with 0.0 confidence if classification fails.
    """
    if _classifier is None:
        return ('government', 0.0)
    
    # Check for human correction first
    corrected = get_correction(news_id)
    if corrected:
        return (corrected, 1.0)  # Human correction has 100% confidence
    
    # Try to get embedding from index
    if _embedding_index and _embeddings is not None:
        row_idx = _embedding_index.get(news_id)
        if row_idx is not None and 0 <= row_idx < len(_embeddings):
            embedding = _embeddings[row_idx].reshape(1, -1)
            
            try:
                prediction = _classifier.predict(embedding)[0]
                probabilities = _classifier.predict_proba(embedding)[0]
                confidence = float(np.max(probabilities))
                
                # Decode label if we have encoder
                if _label_encoder is not None:
                    category = _label_encoder.inverse_transform([prediction])[0]
                elif isinstance(prediction, (int, np.integer)):
                    category = CATEGORIES[prediction] if prediction < len(CATEGORIES) else 'government'
                else:
                    category = str(prediction)
                
                return (category, confidence)
            except Exception as e:
                print(f"[WARN] Classification failed for {news_id}: {e}")
    
    # Fallback: no embedding available
    return ('government', 0.0)


def classify_news(news_items: List[dict]) -> List[dict]:
    """
    Classify a list of news items.
    Adds 'news_id', 'predicted_category', and 'prediction_confidence' to each item.
    
    Human corrections override model predictions.
    """
    corrections = _load_corrections()
    
    for item in news_items:
        source_url = item.get('Source URL', '')
        news_id = generate_news_id(source_url)
        item['news_id'] = news_id
        
        # Check for human correction
        if news_id in corrections:
            item['predicted_category'] = corrections[news_id].get('corrected_label', 'government')
            item['prediction_confidence'] = 1.0
            item['is_corrected'] = True
        else:
            category, confidence = classify_single(news_id, source_url)
            item['predicted_category'] = category
            item['prediction_confidence'] = round(confidence, 3)
            item['is_corrected'] = False
    
    return news_items
