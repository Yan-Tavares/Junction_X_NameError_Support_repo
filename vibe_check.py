# tone_predict.py
import torch
import librosa
import sys
import time
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

# -------------------------
# 3. Predict function
# -------------------------

def classify_intensity(intensity_model, signal, sr=16000):

    inputs = intensity_feature_extractor(signal,
                                         sampling_rate=sr, 
                                         return_tensors="pt", 
                                         padding=True,
                                         max_length=16000*5)
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = intensity_model(**inputs).logits
        # Convert to arousal score (0-1, where higher = more intense)
        instensity_score = torch.softmax(logits, dim=-1).max().item()

    return instensity_score

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

    # Map back to label
    id2label = emotion_model.config.id2label
    return id2label[pred_id]

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

def vibe_check_segment(emotion_model, emotion_feature_extractor, intensity_model, intensity_feature_extractor, file_path, segment, sr=16000):
    # Load audio
    audio, _ = librosa.load(file_path, sr=sr, offset=segment.start, duration=segment.end-segment.start)

    emotion = classify_emotions(emotion_model, audio, sr=sr)
    intensity = classify_intensity(intensity_model, audio, sr=sr)

    return emotion, intensity

# -------------------------
# 4. CLI
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"🎤 Prediction for {file_path}:")

    # -------------------------------
    # Transcribe the audio with Fast Whisper
    # -------------------------------

    segments_list = create_segment_list(file_path)

    # -------------------------------
    # Transcribe the audio with Fast Whisper
    # -------------------------------

    print(f"Processing {len(segments_list)} segments...")
    

    emotion_results = []
    instensity_results = []

    for i, segment in enumerate(segments_list):
        print(f"Processing segment {i+1}/{len(segments_list)}...")
        start = segment.start
        end = segment.end
        duration = end - start
        
        # Load the audio signal once for this segment
        print(f"🔎 Loading file: {file_path}")

        signal, sr = librosa.load(file_path, sr=16000, offset=start, duration=duration)

        emotion, intensity = vibe_check_segment(emotion_model, emotion_feature_extractor, intensity_model, intensity_feature_extractor, segment, sr=16000)

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

    print("Done classifying the audio file!")

    for i, (segment, emotion, intensity) in enumerate(zip(segments_list, emotion_results, instensity_results)):
        print(f"\nSegment {segment.id}: {segment.text}")
        print(f" Emotion: {emotion['emotion']}")
        print(f" Intensity: {intensity['intensity']}")