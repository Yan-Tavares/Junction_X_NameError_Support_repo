import sys
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from faster_whisper import WhisperModel

def extract_prosodic_features(signal, sr=16000):
    """
    Extract prosodic features that indicate speaking tone/style.
    Returns a dictionary with key numerical features.
    """
    features = {}
    
    # 1. Pitch (F0) features - indicates emotional expressiveness
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Only non-zero pitches
            pitch_values.append(pitch)
    
    if len(pitch_values) > 0:
        features['pitch_mean'] = round(np.mean(pitch_values), 2)
        features['pitch_std'] = round(np.std(pitch_values), 2)
        features['pitch_range'] = round(np.max(pitch_values) - np.min(pitch_values), 2)
        features['pitch_variation'] = round(features['pitch_std'] / features['pitch_mean'] if features['pitch_mean'] > 0 else 0, 3)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_range'] = 0
        features['pitch_variation'] = 0
    
    # 2. Energy/Intensity features - indicates emphasis and confidence
    rms = librosa.feature.rms(y=signal)[0]
    features['energy_mean'] = round(np.mean(rms), 4)
    features['energy_std'] = round(np.std(rms), 4)
    features['energy_variation'] = round(features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0, 3)
    
    # 3. Speaking rate proxy - fast (excited/playful) vs slow (serious/disbelief)
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    features['speaking_rate'] = round(np.mean(zcr), 3)
    
    # 4. Spectral features - voice quality (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features['voice_brightness'] = round(np.mean(spectral_centroid), 2)
    
    return features

def categorize_prosodic_features(features):
    """
    Convert numerical prosodic features to categorical labels (high/medium/low).
    """
    categories = {}
    
    # Pitch variation: expressiveness
    if features['pitch_variation'] > 0.3:
        categories['pitch_expressiveness'] = 'high'
    elif features['pitch_variation'] > 0.15:
        categories['pitch_expressiveness'] = 'medium'
    else:
        categories['pitch_expressiveness'] = 'low'
    
    # Energy variation: emphasis
    if features['energy_variation'] > 0.5:
        categories['emphasis'] = 'high'
    elif features['energy_variation'] > 0.3:
        categories['emphasis'] = 'medium'
    else:
        categories['emphasis'] = 'low'
    
    # Speaking rate
    if features['speaking_rate'] > 0.15:
        categories['pace'] = 'fast'
    elif features['speaking_rate'] > 0.08:
        categories['pace'] = 'moderate'
    else:
        categories['pace'] = 'slow'
    
    # Voice brightness
    if features['voice_brightness'] > 2000:
        categories['voice_quality'] = 'bright'
    elif features['voice_brightness'] > 1000:
        categories['voice_quality'] = 'neutral'
    else:
        categories['voice_quality'] = 'warm'
    
    return categories

def transcribe_audio(model, file_path = 'data/first_hateful_speech.wav', word_timestamps = True, printing = False):
    segments, info = model.transcribe(file_path, word_timestamps= word_timestamps)
    # Each segment is an object and it has:
        # id: segment number
        # start: start time (seconds)
        # end: end time (seconds)
        # text: the transcribed text for that segment
        # words: a list of word objects (if word_timestamps=True), each with:
            # word: the word text
            # start: word start time (seconds)
            # end: word end time (seconds)
            # info contains metadata, such as detected language
    
    # Convert generator to list to avoid exhaustion
    segments_list = list(segments)

    if printing:
        print(f"Detected language: {info.language}\n")
        print("Transcription with word-level timestamps:")

        for segment in segments_list:
            print(f"[Segment {segment.id}] {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
            if segment.words:
                for word in segment.words:
                    print(f"    {word.word} ({word.start:.2f}s - {word.end:.2f}s)")

    return segments_list

def create_segment_list(file_path):
    print("Loading Whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Transcribing audio...")

    segments = transcribe_audio(model, file_path=file_path, printing=True)
    segments_list = list(segments)

    return segments_list

def classify_emotions(emotion_model, signal, sr=16000):
    
    # Extract features
    inputs = emotion_feature_extractor(
        signal,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000*5  # 5 seconds max like in training
    )

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Run model
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        instensity_score = torch.softmax(logits, dim=-1).max().item()

    # Map back to label
    id2label = emotion_model.config.id2label

    return id2label[pred_id], instensity_score


def vibe_check_segment(emotion_model, emotion_feature_extractor, intensity_model, intensity_feature_extractor, file_path, segment, sr=16000):
    # Load audio
    audio, _ = librosa.load(file_path, sr=sr, offset=segment.start, duration=segment.end-segment.start)

    emotion, intensity = classify_emotions(emotion_model, audio, sr=sr)
    
    # Extract prosodic features for tone analysis
    prosodic_features = extract_prosodic_features(audio, sr=sr)
    prosodic_categories = categorize_prosodic_features(prosodic_features)

    return emotion, intensity, prosodic_features, prosodic_categories

def augment_segments(segments_list, file_path):
    
    emotion_results = []
    instensity_results = []
    prosodic_results = []
    augmented_texts = {}

    for i, segment in enumerate(segments_list):
        print(f"Processing segment {i+1}/{len(segments_list)}...")
        start = segment.start
        end = segment.end
        duration = end - start
        
        # Load the audio signal once for this segment
        print(f"🔎 Loading file: {file_path}")

        emotion, intensity, prosodic_features, prosodic_categories = vibe_check_segment(
            emotion_model, emotion_feature_extractor, 
            intensity_model, intensity_feature_extractor, 
            file_path, segment, sr=16000
        )

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
        
        prosodic_results.append({
            'id': segment.id,
            'start': start,
            'end': end,
            'text': segment.text,
            'prosodic_features': prosodic_features,
            'prosodic_categories': prosodic_categories
        })

        # Create compact augmented text with categorical labels
        augmented_text = (
            f"{segment.text} "
            f"[emotion={emotion}, emotion confidence={intensity:.2f}, "
            f"pitch={prosodic_categories['pitch_expressiveness']}, "
            f"emphasis={prosodic_categories['emphasis']}, "
            f"pace={prosodic_categories['pace']}]"
        )
        augmented_texts[segment.id] = augmented_text
    
    # Save to json file
    with open("data/augmented_texts.json", "w") as f:
        import json
        json.dump(augmented_texts, f, indent=4)

    print("Done classifying the audio file!")
    return emotion_results, instensity_results, prosodic_results, augmented_texts

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"🎤 Prediction for {file_path}:")

    segments_list = create_segment_list(file_path)
    print(f"Processing {len(segments_list)} segments...")

    emotion_results, instensity_results, prosodic_results, augmented_texts = augment_segments(segments_list, file_path)

    for i, segment in enumerate(segments_list):
        emotion = emotion_results[i]
        intensity = instensity_results[i]
        prosodic = prosodic_results[i]
        augmented_text = augmented_texts[segment.id]
        
        print(f"\n{'='*80}")
        print(f"Segment {segment.id}: {segment.text}")
        print(f"Time: {segment.start:.2f}s - {segment.end:.2f}s")
        print(f"Augmented Text:\n{augmented_text}")
        print(f"{'='*80}")