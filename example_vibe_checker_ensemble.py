# Example: Using Vibe_Checker_model with Ensemble

from transformers import WhisperModel
from segment_augmentation import create_segment_list
from src.model.ensemble import Ensemble
from Vibe_Checker_Class import Vibe_Checker_model


# Example 1: Using Vibe_Checker as standalone
print("=" * 80)
print("EXAMPLE 1: Standalone Usage")
print("=" * 80)

vibe_checker = Vibe_Checker_model()

file_path = 'data/youtube_audio.wav'
segments = create_segment_list(file_path)

predictions = vibe_checker.predict(segments, audio_path=file_path)
print(f"\nPredictions shape: {predictions.shape}")
print(f"Predictions:\n{predictions}")
print("\nLabel format: [p_normal, p_offensive, p_extremist]")

# Example 2: Using in Ensemble with other models
print("\n" + "=" * 80)
print("EXAMPLE 2: Ensemble Usage")
print("=" * 80)

# Import your other models
# from your_text_model import YourTextModel
# from your_audio_model import YourAudioModel

# Create ensemble with multiple models
models = [
    vibe_checker,  # LLM-based with prosodic features
    # YourTextModel(),  # Add your text-based models
    # YourAudioModel(), # Add your audio-based models
]

ensemble = Ensemble(models=models, whisper_size="base")

# Run ensemble prediction on audio file
audio_path = "data/youtube_audio.wav"
ensemble_predictions = ensemble.predict(audio_path)

print(f"\nEnsemble predictions shape: {ensemble_predictions.shape}")
print(f"Sample predictions:\n{ensemble_predictions[:3]}")

# Assemble into API-compatible format
results = ensemble.assemble_preds(ensemble_predictions)

print(f"\n" + "=" * 80)
print("Assembled Results:")
print("=" * 80)
print(f"Audio path: {results['audio_path']}")
print(f"Transcript: {results['transcript'][:100]}...")
print(f"\nDetected hate spans: {len(results['hate_spans'])}")

for span in results['hate_spans'][:3]:  # Show first 3
    print(f"\n  [{span['start']:.2f}s - {span['end']:.2f}s]")
    print(f"  Text: {span['text']}")
    print(f"  Label: {span['label']}")
    print(f"  Confidence: {span['confidence']:.3f}")
