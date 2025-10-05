"""
SentimentModel - Text-based hate speech classifier
Wraps the HateXplain fine-tuned model for ensemble integration
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SentimentModel:
    """
    Text-based sentiment/hate speech classification model.
    Uses a fine-tuned transformer model to classify text into:
    - normal (non-hate)
    - offensive (potentially problematic)
    - extremist (hate speech)
    """
    
    def __init__(self, text_model_path="cardiffnlp/twitter-roberta-base-hate-latest", **kwargs):
        """
        Initialize the sentiment model.
        
        Args:
            text_model_path: HuggingFace model ID or local path
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.input_type = "text"
        self.text_model_path = text_model_path
        
        print(f"Loading sentiment model from {text_model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                text_model_path
            ).to(DEVICE).eval()
            
            print(f"✅ Sentiment model loaded on {DEVICE}")
        except Exception as e:
            print(f"❌ Failed to load sentiment model: {e}")
            raise
    
    def predict(self, texts):
        """
        Predict hate speech probabilities for text segments.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            numpy.ndarray: Probabilities of shape (n_texts, 3)
                          [p_normal, p_offensive, p_extremist]
        """
        if not texts:
            return np.zeros((0, 3), dtype=np.float32)
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Check if binary or 3-class model
        num_classes = probs.shape[1]
        
        if num_classes == 2:
            # Binary classification model (non-hate, hate)
            # Convert to 3-class format: [normal, offensive, extremist]
            print("⚠️ Model has 2 classes, mapping to 3 classes")
            
            three_class_probs = np.zeros((probs.shape[0], 3), dtype=np.float32)
            
            # probs[:, 0] = p(non-hate)
            # probs[:, 1] = p(hate)
            
            # Map based on confidence:
            # - High confidence non-hate (>0.7) -> [1.0, 0.0, 0.0] (normal)
            # - Low confidence non-hate (0.3-0.7) -> [p_non_hate, 0.0, p_hate] split between normal/extremist
            # - Uncertain (0.3-0.7) -> put probability in middle class (offensive)
            # - High confidence hate (>0.7) -> [0.0, 0.0, 1.0] (extremist)
            
            p_non_hate = probs[:, 0]
            p_hate = probs[:, 1]
            
            # Vectorized approach: create masks for each case
            high_hate_mask = p_hate >= 0.70      # High confidence hate
            high_non_hate_mask = p_hate <= 0.30  # High confidence non-hate
            uncertain_mask = ~(high_hate_mask | high_non_hate_mask)  # Everything else
            
            # Case 1: High confidence hate -> extremist
            three_class_probs[high_hate_mask, 0] = 1.0 - p_hate[high_hate_mask]  # normal
            three_class_probs[high_hate_mask, 1] = 0.0                           # offensive
            three_class_probs[high_hate_mask, 2] = p_hate[high_hate_mask]        # extremist
            
            # Case 2: High confidence non-hate -> normal
            three_class_probs[high_non_hate_mask, 0] = p_non_hate[high_non_hate_mask]      # normal
            three_class_probs[high_non_hate_mask, 1] = 1.0 - p_non_hate[high_non_hate_mask] # offensive
            three_class_probs[high_non_hate_mask, 2] = 0.0                                  # extremist
            
            # Case 3: Uncertain -> distribute to offensive class
            three_class_probs[uncertain_mask, 0] = p_non_hate[uncertain_mask] * 0.3  # normal
            three_class_probs[uncertain_mask, 1] = 0.5                               # offensive
            three_class_probs[uncertain_mask, 2] = p_hate[uncertain_mask] * 0.3      # extremist
            
            # Normalize to ensure sum=1
            three_class_probs = three_class_probs / (three_class_probs.sum(axis=1, keepdims=True) + 1e-9)
            return three_class_probs
        
        elif num_classes == 3:
            # Already 3-class model, apply weights
            weights = np.array([1.3, 1.1, 1.0])  # [normal, offensive, extremist]
            weighted_probs = probs * weights
            weighted_probs = weighted_probs / weighted_probs.sum(axis=1, keepdims=True)
            return weighted_probs.astype(np.float32)
        
        else:
            # Unexpected number of classes
            print(f"⚠️ Unexpected number of classes: {num_classes}, padding/truncating to 3")
            result = np.zeros((probs.shape[0], 3), dtype=np.float32)
            min_classes = min(num_classes, 3)
            result[:, :min_classes] = probs[:, :min_classes]
            # Normalize
            result = result / (result.sum(axis=1, keepdims=True) + 1e-9)
            return result
