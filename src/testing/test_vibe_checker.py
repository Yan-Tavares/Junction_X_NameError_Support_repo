"""
Test suite for VibeCheckerModel
Tests prosodic feature extraction, emotion classification, and LLM integration
"""

import sys
import os

# Force unbuffered output for real-time printing
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import librosa
import torch
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# os.path.join(..., "..", "..") -> Goes up one directories from testing
# os.path.abspath(...) -> Converts that path to an absolute path
# sys.path.append(...) -> Adds the project root directory to Python's module search path.

from src.model.vibechecker import VibeCheckerModel


def test_model_initialization(device, tuned_emotion_model_dir):
    """Test 1: Model loads successfully"""
    print("\n" + "="*80, flush=True)
    print("TEST 1: Model Initialization", flush=True)
    print("="*80, flush=True)
    
    try:
        print('Trying to load emotion model from directory:', os.path.abspath(tuned_emotion_model_dir), flush=True)
        model = VibeCheckerModel(device = device, tuned_emotion_model_dir=tuned_emotion_model_dir)

        # Check attributes
        assert hasattr(model, 'input_type'), "Model should have input_type attribute"
        assert model.input_type == "audio", "input_type should be 'audio'"
        assert hasattr(model, 'emotion_model'), "Model should have emotion_model"
        assert hasattr(model, 'emotion_feature_extractor'), "Model should have feature extractor"
        
        print("✅ PASSED: Model initialized successfully")
        print(f"   - Device: {model.device}")
        print(f"   - Input type: {model.input_type}")
        return model
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise


def test_transcription(model, audio_path):
    """Test 1.5: Transcription works"""
    print("\n" + "="*80)
    print("TEST 1.5: Transcription")
    print("="*80)
        
    try:
        segments = model.transcribe_audio(audio_path)


        # Assert if the first segment has the attributes .text, .start, .end and .audio_segment
        assert hasattr(segments[0], 'text'), "Segment should have 'text' attribute"
        assert hasattr(segments[0], 'start'), "Segment should have 'start' attribute"
        assert hasattr(segments[0], 'end'), "Segment should have 'end' attribute"
        assert hasattr(segments[0], 'audio_segment'), "Segment should have 'audio_segment' attribute"

        print("✅ PASSED: Transcription and audio splitting works")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        raise

    return segments


def test_prosodic_features(model, segments, segment_num):
    """Test 2: Prosodic feature extraction"""
    print("\n" + "="*80)
    print("TEST 2: Prosodic Feature Extraction")
    print("="*80)
    
    try:
        segment = segments[segment_num] 
        
        print(f'\n🎵 Testing segment number {segment_num}')
        print(f'   - Start: {segment.start:.2f}s')
        print(f'   - End: {segment.end:.2f}s')
        print(f'   - Text: {segment.text}')

        signal = segment.audio_segment

        sr = 16000
        features = model.extract_prosodic_features(signal, sr=sr)
        categories = model.categorize_prosodic_features(features)


        # Validate feature keys
        required_keys = ['pitch_variation', 'energy_variation', 'speaking_rate']
        for key in required_keys:
            assert key in features, f"Missing feature: {key}"
        
        # Validate category keys
        required_categories = ['pitch', 'emphasis', 'pace']
        for key in required_categories:
            assert key in categories, f"Missing category: {key}"

        
        print("\n📉 Monotone speech (low variation):")
        print(f"   - Pitch variation: {features['pitch_variation']:.3f}")
        print(f"   - Energy variation: {features['energy_variation']:.3f}")
        print(f"   - Speaking rate: {features['speaking_rate']:.3f}")
        print(f"   - Categories: Pitch={categories['pitch']}, Emphasis={categories['emphasis']}, Pace={categories['pace']}")

        print("\n✅ PASSED: Prosodic features extracted successfully")
        return True
    
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        raise


def test_emotion_classification(model, audio_path=None):
    """Test 3: Emotion classification"""
    print("\n" + "="*80)
    print("TEST 3: Emotion Classification")
    print("="*80)
    
    try:
        if audio_path and os.path.exists(audio_path):
            # Use real audio file
            print(f"\n🎵 Testing with real audio: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000, duration=5)
        else:
            # Generate synthetic audio
            print("\n🎵 Testing with synthetic audio")
            sr = 16000
            duration = 3
            t = np.linspace(0, duration, sr * duration)
            audio = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
        
        emotion, confidence = model.classify_emotion(audio, sr=sr)
        
        print(f"\n🎭 Emotion Detection Results:")
        print(f"   - Emotion: {emotion}")
        print(f"   - Confidence: {confidence:.3f}")
        
        # Validate outputs
        assert isinstance(emotion, str), "Emotion should be a string"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        
        print("\n✅ PASSED: Emotion classification working")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        raise


def test_segment_prediction(model, segments, audio_path, max_predictions=None):
    """Test 5: Predict on audio file"""
    print("\n" + "="*80)
    print("TEST 5: Prediction with Audio")
    print("="*80)

    start_time = time.time()

    if max_predictions is not None:
        if len(segments) > max_predictions:
            segments = segments[:max_predictions]  # Limit to first N segments for testing
        
    try:
        print(f"   - Found {len(segments)} segments")
        predictions = model.predict(segments, audio_path=audio_path)
        
        print(f"\n📊 Prediction Results:")
        print(f"   - Shape: {predictions.shape}")
        
        for i, (seg, pred) in enumerate(zip(segments, predictions)):
            print(f"\n   Segment {i+1}: '{seg.augmented_text}'")
            print(f"     [p_normal={pred[0]:.3f}, p_offensive={pred[1]:.3f}, p_extremist={pred[2]:.3f}]")
            
            # Determine classification
            label_idx = np.argmax(pred)
            labels = ['normal', 'offensive', 'extremist']
            print(f"      → Classified as: {labels[label_idx]} (confidence: {pred[label_idx]:.3f})")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n⏱️  Time taken for predictions: {elapsed_time:.2f} seconds")

        # Validate predictions
        assert predictions.shape == (len(segments), 3), f"Wrong shape: {predictions.shape}"
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=0.01), "Probabilities should sum to 1"
        
        print("\n✅ PASSED: Real audio predictions successful")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(audio_path, tuned_emotion_model_dir = 'fine_tuned_emotion_model'):
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 VIBE CHECKER MODEL TEST SUITE")
    print("="*80)
    
    # Confirm that the audio file exists before running tests
    if not os.path.exists(audio_path):
        print(f"❌ ERROR: The audio for testing file '{audio_path}' does not exist. Please provide a valid audio file path.")
        sys.exit(1)
    
    test_results = []
    
    try:
        # Test 1: Initialization
        model = test_model_initialization('cuda', tuned_emotion_model_dir)
        test_results.append(("Initialization", True))

        # Test 2: Transcription
        segments = test_transcription(model, audio_path)
        
        # Test 3: Prosodic features
        test_prosodic_features(model, segments, segment_num=0)
        test_results.append(("Prosodic Features", True))

        # Test 4: Emotion classification
        test_emotion_classification(model, audio_path)
        test_results.append(("Emotion Classification", True))

        # Test 5: Real audio predictions
        result = test_segment_prediction(model, segments, audio_path, max_predictions=25)
        test_results.append(("Real Audio Predictions", result))

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*80}\n")

    success = (passed == total)

    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💔 Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VibeCheckerModel")
    parser.add_argument(
        "--audio",
        type=str,
        default="src/testing/test_extremist_audio.wav",
        help="Path to audio file for testing"
    )

    args = parser.parse_args()
    success = run_all_tests(args.audio, tuned_emotion_model_dir="fine_tuned_emotion_model")

    
