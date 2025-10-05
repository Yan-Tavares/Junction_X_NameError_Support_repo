import sys
import torch
import librosa
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from faster_whisper import WhisperModel
from GPT_api import analyze_extremism
import json
import os


def LLM_clf_w_audio_context(json_path):
    """
    Processes augmented texts from JSON and analyzes them with LLM.
    The augmented text format is:
    "Text here [emotion=neutral, intensity=0.42, pitch=high, emphasis=high, pace=moderate]"
    """
    llm_results = []
    
    with open(json_path, "r", encoding="utf-8") as f:
        augmented_texts_dict = json.load(f)

    # Process each augmented text
    for segment_id, augmented_text in augmented_texts_dict.items():
        print(f"Processing segment {segment_id}...")
        print(f"Augmented text: {augmented_text}")
        
        # 🔥 LLM analysis with full augmented context (includes prosodic features)
        # The augmented text already contains emotion, intensity, and prosodic labels
        llm_result = analyze_extremism(augmented_text)
        
        llm_results.append({
            'id': segment_id,
            'augmented_text': augmented_text,
            'extremism_analysis': llm_result
        })

    print("Done with LLM classification!")
    return llm_results


if __name__ == "__main__":

    json_path = 'data/augmented_texts.json'
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        print("Please run LLM_clf_w_audio_context.py first to generate the augmented texts.")
        sys.exit(1)
    
    print(f"🎤 Loading augmented texts from: {json_path}\n")
    
    llm_results = LLM_clf_w_audio_context(json_path)

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    for llm in llm_results:
        print(f"\n{'='*80}")
        print(f"Segment {llm['id']}")
        print(f"Augmented Text:\n{llm['augmented_text']}")
        print(f"Extremism Analysis:\n{llm['extremism_analysis']}")
        print(f"{'='*80}")