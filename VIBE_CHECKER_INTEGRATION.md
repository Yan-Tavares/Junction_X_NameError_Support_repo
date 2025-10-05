# Vibe_Checker_model Integration Guide

## Overview
The `Vibe_Checker_model` is now fully compatible with the `Ensemble` class architecture. It uses LLM-based analysis with prosodic features to detect extremist content.

## Architecture

### Class Structure
```python
class Vibe_Checker_model:
    input_type = "audio"  # Required by ensemble
    
    def __init__(self, device=None)
    def predict(self, segments, audio_path=None) -> np.ndarray
```

### How It Fits in Ensemble

The `Ensemble` class (from `ensemble.py`) expects models with:
1. **`input_type`** attribute: `"audio"` or `"text"`
2. **`predict()`** method returning: `(n_segments, n_labels)` array

## Prediction Pipeline

### 1. Input Format
Segments from transcription:
```python
segments = [
    {
        'start': 0.0,
        'end': 3.5,
        'text': 'Transcribed text here',
        'audio_path': 'path/to/audio.wav'  # Optional if passed separately
    },
    ...
]
```

### 2. Processing Steps

For each segment:
1. **Load Audio**: Extract audio chunk from `start` to `end`
2. **Emotion Detection**: Use fine-tuned emotion model
3. **Prosodic Analysis**: Extract pitch, emphasis, pace features
4. **Augment Text**: Create enriched text with audio context
5. **LLM Analysis**: Classify extremism using augmented text
6. **Convert to Probabilities**: Map LLM result to 3-class distribution

### 3. Output Format

Returns `(n_segments, 3)` array:
```python
[
    [p_normal, p_offensive, p_extremist],  # Segment 1
    [p_normal, p_offensive, p_extremist],  # Segment 2
    ...
]
```

## Augmented Text Format

The model creates enriched context for the LLM:
```
"Original transcribed text [emotion=angry, emotion confidence=0.85, pitch=high, emphasis=high, pace=fast]"
```

This prosodic context helps the LLM understand:
- **Sarcasm**: high pitch + high emphasis + neutral emotion
- **Genuine anger**: high pitch + high emphasis + angry emotion
- **Disbelief**: slow pace + questioning tone
- **Playfulness**: high emphasis + fast pace

## Probability Mapping

### When LLM says "Extremist":
```python
predictions = [
    0.1 * (1 - confidence),      # p_normal (low)
    0.3 * (1 - confidence),      # p_offensive (moderate)
    0.6 + 0.4 * confidence       # p_extremist (high)
]
```

### When LLM says "Not Extremist":
```python
predictions = [
    0.7 * (1 - confidence) + 0.3,  # p_normal (high)
    0.3 * confidence,              # p_offensive (low-moderate)
    0.0                            # p_extremist (none)
]
```

## Usage Examples

### Standalone Usage
```python
from LLM_vibe_checker import Vibe_Checker_model

vibe_checker = Vibe_Checker_model()

segments = [...]  # From transcription
predictions = vibe_checker.predict(segments, audio_path='audio.wav')
```

### Ensemble Usage
```python
from src.model.ensemble import Ensemble
from LLM_vibe_checker import Vibe_Checker_model

# Create ensemble with Vibe_Checker + other models
models = [
    Vibe_Checker_model(),
    # YourTextModel(),
    # YourAudioModel(),
]

ensemble = Ensemble(models=models, whisper_size="base")
predictions = ensemble.predict("audio.wav")
results = ensemble.assemble_preds(predictions)
```

## Key Features

### 1. Prosodic-Aware Classification
- Considers HOW something is said, not just WHAT is said
- Detects sarcasm, tone, emphasis patterns
- More nuanced than text-only analysis

### 2. LLM-Powered Intelligence
- Uses llama3.2 (via Ollama) for sophisticated reasoning
- Understands context and cultural nuances
- Can detect implicit extremism

### 3. Ensemble Compatible
- Works seamlessly with other models
- Predictions are automatically weighted and combined
- Follows standard architecture patterns

## Error Handling

The model gracefully handles errors:
- **Audio loading fails**: Returns neutral prediction `[0.7, 0.2, 0.1]`
- **LLM fails**: Returns cautious default `[0.7, 0.2, 0.1]`
- **No text**: Returns normal `[1.0, 0.0, 0.0]`
- **No audio path**: Returns all normal predictions

## Requirements

### Models Needed:
1. **Emotion Model**: `./fine_tuned_emotion_model/`
2. **LLM**: llama3.2 via Ollama (run `ollama pull llama3.2`)

### Dependencies:
```python
torch
librosa
numpy
transformers
faster-whisper
langchain_ollama
trustcall
```

## Performance Considerations

### Speed:
- **~5-10 seconds per segment** (LLM processing time)
- For faster processing, use ensemble weights to reduce Vibe_Checker influence
- Consider batch processing or caching for repeated analysis

### Accuracy:
- **Strengths**: Excellent at detecting tone-based extremism (sarcasm, mockery)
- **Weaknesses**: Slower than pure ML models
- **Best used**: As one model in an ensemble for comprehensive coverage

## Ensemble Weight Tuning

Adjust model importance:
```python
ensemble.weights = np.array([
    0.4,  # Vibe_Checker (slower but smarter)
    0.3,  # Text model (fast, keyword-based)
    0.3,  # Audio model (fast, prosody-based)
])
```

## Future Enhancements

Possible improvements:
1. **Batch Processing**: Analyze multiple segments in parallel
2. **Caching**: Store LLM results for repeated audio
3. **Confidence Thresholds**: Skip LLM for obviously normal segments
4. **Model Selection**: Use smaller/faster LLMs for less critical analysis
5. **Feature Reduction**: Only extract most important prosodic features

## Troubleshooting

### "model 'llama3.2' not found"
```bash
ollama pull llama3.2
```

### "fine_tuned_emotion_model not found"
- Ensure the model directory exists
- Train or download the emotion model first

### Slow predictions
- Reduce ensemble weight for Vibe_Checker
- Use faster LLM (qwen2.5:3b)
- Add confidence thresholds to skip obvious cases

### LLM connection errors
```bash
# Check if Ollama is running
ollama list

# Restart Ollama if needed
ollama serve
```
