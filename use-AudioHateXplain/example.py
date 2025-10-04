"""
Simple example of using the AudioHateXplain pipeline
"""

from pathlib import Path
from run_AudioHateXplain import run_analysis

def main():
    # Get the project root directory (parent of use-AudioHateXplain)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Configure paths using absolute paths
    input_audio = project_root / "data" / "recordings" / "trump.wav"
    output_dir = project_root / "output" / "analysis"
    
    # Convert to strings
    input_audio = str(input_audio)
    output_dir = str(output_dir)
    
    print("Starting AudioHateXplain analysis...")
    print(f"Input: {input_audio}")
    print(f"Output: {output_dir}/analysis.json")
    
    # Check if input file exists
    if not Path(input_audio).exists():
        print(f"\n❌ Error: Input file not found at: {input_audio}")
        print(f"   Please make sure the file exists or update the path in this script.")
        return None
    
    print()
    
    # Run the analysis
    results = run_analysis(
        input_path=input_audio,
        output_path=output_dir
    )
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total segments analyzed: {results['summary']['total_segments']}")
    print(f"Hate speech detected: {results['summary']['hate_segments']} segments")
    print(f"Offensive content: {results['summary']['offensive_segments']} segments")
    print(f"Normal content: {results['summary']['normal_segments']} segments")
    print(f"Uncertain: {results['summary']['uncertain_segments']} segments")
    print(f"\nHate speech percentage: {results['summary']['hate_percentage']:.2f}%")
    print("="*70)
    
    # Show flagged segments
    if results['summary']['hate_segments'] > 0 or results['summary']['offensive_segments'] > 0:
        print("\n⚠️  FLAGGED CONTENT:")
        for segment in results['segments']:
            if segment['label'] in ['hate', 'offensive']:
                print(f"\n  [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s]")
                print(f"  Label: {segment['label'].upper()}")
                print(f"  Confidence: {segment['confidence']:.2%}")
                print(f"  Text: \"{segment['text']}\"")
    
    return results


if __name__ == "__main__":
    main()
