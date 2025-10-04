
from vibe_check import vibe_check_segment, create_segment_list
import sys
import torch
import librosa
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from faster_whisper import WhisperModel

# -------------------------
# 1. Device
# -------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print('MPS')
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# -------------------------
# 2. Load fine-tuned model
# -------------------------
MODEL_DIR = "./fine_tuned_emotion_model"
emotion_model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR).to(DEVICE)
emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)


model_name = "superb/wav2vec2-base-superb-er"
intensity_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(DEVICE)
intensity_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
print("Intensity model loaded successfully!")



def LLM_clf_w_audio_context(segments_list, file_path):
    
    emotion_results = []
    instensity_results = []

    for i, segment in enumerate(segments_list):
        print(f"Processing segment {i+1}/{len(segments_list)}...")
        start = segment.start
        end = segment.end
        duration = end - start
        
        # Load the audio signal once for this segment
        print(f"🔎 Loading file: {file_path}")

        emotion, intensity = vibe_check_segment(emotion_model, emotion_feature_extractor, intensity_model, intensity_feature_extractor, file_path, segment, sr=16000)

        emotion_results.append({
            'id': segment.id,
            'start': start,
            'end': end,
            'text': segment.text,
            'emotion': emotion
        })

        instensity_results.append({
            'id': segment.id,
            'start': start,
            'end': end,
            'text': segment.text,
            'intensity': intensity
        })

        # Add emotion and rounded intensity as a string to the segment text
        segment.text += f" [Emotion: {emotion}, Intensity: {intensity:.1f}]"

    print("Done classifying the audio file!")
    return emotion_results, instensity_results, segments_list


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"🎤 Prediction for {file_path}:")

    segments_list = create_segment_list(file_path)
    print(f"Processing {len(segments_list)} segments...")

    emotion_results, instensity_results, segments_list = LLM_clf_w_audio_context(segments_list, file_path)

    for i, (segment, emotion, intensity) in enumerate(zip(segments_list, emotion_results, instensity_results)):
        print(f"\nSegment {segment.id}: {segment.text}")
        print(f" Emotion: {emotion['emotion']}")
        print(f" Intensity: {intensity['intensity']}")