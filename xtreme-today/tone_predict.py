# tone_predict.py
import torch
import librosa
import sys
import time
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

# -------------------------
# 1. Device
# -------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# -------------------------
# 2. Load fine-tuned model
# -------------------------
MODEL_DIR = "./fine_tuned_emotion_model"
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR).to(DEVICE)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)

# -------------------------
# 3. Predict function
# -------------------------
def predict(file_path):
    start_time = time.time()
    print(f"🔎 Loading file: {file_path}")

    # Load audio
    audio, _ = librosa.load(file_path, sr=16000)

    # Extract features
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000*5  # 5 seconds max like in training
    )

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Run model
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    # Map back to label
    id2label = model.config.id2label
    elapsed = time.time() - start_time
    print(f"⏱️ Inference took {elapsed:.2f} seconds")
    return id2label[pred_id]

# -------------------------
# 4. CLI
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tone_predict.py path/to/file.wav")
        sys.exit(1)

    file_path = sys.argv[1]
    label = predict(file_path)
    print(f"🎤 Prediction for {file_path}: {label}")
