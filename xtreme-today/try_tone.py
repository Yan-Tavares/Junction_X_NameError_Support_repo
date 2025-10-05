import argparse
from pathlib import Path
from tone_utils import to_wav, chunk_audio, load_tone_model, analyze_audio_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze emotions in audio file')
    parser.add_argument('audio_file', help='Path to the audio file to analyze')
    parser.add_argument('--chunk-size', type=int, default=10,
                      help='Size of audio chunks in seconds (default: 10)')
    args = parser.parse_args()

    # Convert to proper Path object and verify file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        # Try with parent directory path
        parent_path = Path('/Users/inesmarques/Desktop/Junction_X_NameError_Support_repo')
        alt_path = parent_path / 'examples' / audio_path.name
        if alt_path.exists():
            audio_path = alt_path
        else:
            print(f"Error: File not found at either:")
            print(f"  {audio_path}")
            print(f"  {alt_path}")
            return

    # Process audio file
    print(f"Analyzing: {audio_path}")
    feature_extractor, model = load_tone_model()
    results = analyze_audio_file(str(audio_path), feature_extractor, model, 
                               chunk_size_s=args.chunk_size)

    # Print results
    print("\nEmotion Analysis Results:")
    print("-" * 30)
    for timestamp, emotion in results:
        print(f"{timestamp:.1f}s: {emotion}")

if __name__ == "__main__":
    main()