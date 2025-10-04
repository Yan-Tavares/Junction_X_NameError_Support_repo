# AudioHateXplain Runner

This script processes audio files through an ASR (Automatic Speech Recognition) and hate speech detection pipeline.

## Features

- 🎤 Transcribes audio with timestamps
- 🔍 Detects hate speech in transcribed segments
- 📊 Provides confidence scores and labels
- 💾 Outputs structured JSON analysis

## Installation

Install the required dependencies:

```bash
pip install torch torchaudio transformers safetensors
```

Or use the project requirements if available.

## Usage

### Basic Usage

```python
from run_AudioHateXplain import run_analysis

# Process an audio file
results = run_analysis(
    input_path="path/to/audio.wav",
    output_path="path/to/output/directory"
)
```

### Advanced Usage

```python
from run_AudioHateXplain import AudioHateXplainPipeline

# Initialize pipeline with custom models
pipeline = AudioHateXplainPipeline(
    asr_model_path="models/AudioHateXplain/ASR",
    classifier_model_path="models/AudioHateXplain/classifier",  # Optional
    device="cuda"  # or "cpu"
)

# Process audio file
results = pipeline.process_audio_file(
    input_path="data/recordings/test.wav",
    output_path="output/analysis"
)
```

## Output Format

The script generates an `analysis.json` file with the following structure:

```json
{
  "audio_file": "test.wav",
  "total_segments": 5,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.5,
      "text": "This is the transcribed text",
      "label": "normal",
      "confidence": 0.9234,
      "scores": {
        "normal": 0.9234,
        "offensive": 0.0500,
        "hate": 0.0266
      }
    }
  ],
  "summary": {
    "total_segments": 5,
    "hate_segments": 0,
    "offensive_segments": 1,
    "normal_segments": 4,
    "uncertain_segments": 0,
    "hate_percentage": 0.0
  }
}
```

## Labels

The classifier outputs one of the following labels:

- **`normal`**: No hate speech or offensive content detected
- **`offensive`**: Offensive language detected (but not hate speech)
- **`hate`**: Hate speech detected
- **`uncertain`**: Low confidence prediction (< 0.5)

## Model Details

### ASR Model

- **Location**: `models/AudioHateXplain/ASR/model.safetensors`
- **Format**: SafeTensors (automatically loaded)
- **Fallback**: If custom model fails, uses `openai/whisper-tiny.en`

### Hate Speech Classifier

- **Default**: Uses BERT-base-uncased (3-class classification)
- **Can be customized**: Pass path to `classifier_model_path`
- **Expected**: HuggingFace transformer model with 3 labels

## Configuration

### Audio Chunking

The script processes audio in 30-second chunks by default. Adjust in the code:

```python
chunk_duration = 30  # seconds (line ~141)
```

### Confidence Threshold

Labels below 50% confidence are marked as "uncertain". Adjust:

```python
if confidence < 0.5:  # Change threshold here (line ~265)
    label = "uncertain"
```

## Troubleshooting

### Model Loading Issues

If the custom ASR model fails to load:
- ✅ The script automatically falls back to Whisper
- Check that `model.safetensors` exists in the correct path
- Ensure `config.json` is present alongside the weights

### CUDA/GPU Issues

If you get CUDA errors:
```python
pipeline = AudioHateXplainPipeline(device="cpu")
```

### Audio Format Issues

The script expects `.wav` files. Convert other formats:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Example

```python
# Simple example
from run_AudioHateXplain import run_analysis

results = run_analysis(
    input_path="data/recordings/test.wav",
    output_path="output"
)

print(f"Detected {results['summary']['hate_segments']} hate speech segments")
```

## Notes

- The ASR model transcribes in ~30-second segments for better timestamp granularity
- Each segment is classified independently
- Confidence scores are based on softmax probabilities
- The script handles stereo-to-mono conversion automatically
- Audio is automatically resampled to 16kHz if needed

## License

See project LICENSE file.
