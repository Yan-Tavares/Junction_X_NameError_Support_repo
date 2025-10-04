import warnings
warnings.filterwarnings("ignore")

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["SPEECHBRAIN_DISABLE_SYMLINKS"] = "1"

# Only keep lightweight imports at the top for immediate startup
import numpy as np
import torch
import librosa
from pydub import AudioSegment
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel

# Heavy imports moved inside functions for lazy loading:
# - faster_whisper, torch, torchaudio, librosa, pydub, transformers
# This makes the script start much faster!

# With this:

# ffmpeg required, install via choco
# sympy-1.13.1
# torch = torch-2.5.1+cu121
# torchaudio-2.5.1+cu121

# Use faster-whisper for word-level timestamps

def download_youtube_audio(youtube_url, output_path='data/youtube_audio.m4a'):
    """
    Downloads audio from a YouTube video and saves it as M4A format.
    
    Args:
        youtube_url: The YouTube video URL
        output_path: Where to save the downloaded audio (default: 'data/youtube_audio.m4a')
    
    Returns:
        The path to the downloaded audio file
    """

    from yt_dlp import YoutubeDL
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.m4a', '').replace('.mp4', ''),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': False,
        'no_warnings': False,
    }
    
    print(f"Downloading audio from: {youtube_url}")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    print(f"Audio downloaded to: {output_path}")
    return output_path

def mp4_to_wav(mp4_path, wav_path):
    """
    Converts an MP4/M4A audio/video file to WAV format using pydub (requires ffmpeg).
    """
    
    audio = AudioSegment.from_file(mp4_path)
    audio.export(wav_path, format="wav")

def classify_emotions(wav_path, segments):
    """
    Classifies the emotion/tone of each segment using a pretrained emotion model.
    Returns a list of (segment_id, start, end, text, emotion_label).
    """
    import torch
    import librosa
    
    print("Loading emotion classifier...")
    
    try:
        # Use HuggingFace transformers pipeline - more reliable and compatible
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        import torch.nn.functional as F
        
        # Load a working emotion recognition model
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        
        # Emotion labels for this model
        emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        
        model_loaded = True
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to neutral classification...")
        model_loaded = False
    
    results = []
    segments_list = list(segments)  # Convert generator to list
    print(f"Processing {len(segments_list)} segments...")
    
    for i, segment in enumerate(segments_list):
        print(f"Processing segment {i+1}/{len(segments_list)}...")
        start = segment.start
        end = segment.end
        duration = end - start
        
        try:
            if model_loaded:
                # Load segment audio using librosa at 16kHz
                signal, sr = librosa.load(wav_path, sr=16000, offset=start, duration=duration)
                
                # Extract features
                inputs = feature_extractor(signal, sampling_rate=16000, return_tensors="pt", padding=True)
                
                # Get prediction
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_id = torch.argmax(logits, dim=-1).item()
                    emotion = emotion_labels[predicted_id]
            else:
                emotion = "neutral"
                
        except Exception as e:
            print(f"Error processing segment {i+1}: {e}")
            emotion = "unknown"
        
        results.append({
            'id': segment.id,
            'start': start,
            'end': end,
            'text': segment.text,
            'emotion': emotion
        })
    
    print("Done classifying!")
    return results

def classify_intensity_and_arousal(wav_path, segments):
    """
    Classifies the intensity/arousal level of speech segments.
    Detects things like: calm, normal, intense, agitated, arousal level
    Returns a list with arousal/intensity scores.
    """

    print("Loading intensity/arousal classifier...")
    
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        import numpy as np
        
        # Use a model that can detect arousal/activation level
        # This model detects emotional dimensions including arousal (calm vs excited/intense)
        model_name = "superb/wav2vec2-base-superb-er"  # Emotion recognition with dimensional output
        
        try:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            model_loaded = True
            print("Arousal model loaded successfully!")
        except:
            print("Using alternative approach - analyzing audio features for intensity...")
            model_loaded = False
        
    except Exception as e:
        print(f"Error loading arousal model: {e}")
        model_loaded = False
    
    results = []
    segments_list = list(segments)
    print(f"Processing {len(segments_list)} segments for intensity/arousal...")
    
    for i, segment in enumerate(segments_list):
        print(f"Processing segment {i+1}/{len(segments_list)} for intensity...")
        start = segment.start
        end = segment.end
        duration = end - start
        
        try:
            # Load audio segment
            signal, sr = librosa.load(wav_path, sr=16000, offset=start, duration=duration)
            
            if model_loaded:
                # Use model prediction
                inputs = feature_extractor(signal, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    # Convert to arousal score (0-1, where higher = more intense)
                    arousal_score = torch.softmax(logits, dim=-1).max().item()
            else:
                # Fallback: Calculate intensity from audio features
                # RMS energy (volume/intensity)
                rms = librosa.feature.rms(y=signal)[0]
                avg_rms = np.mean(rms)
                
                # Zero crossing rate (higher for more agitated speech)
                zcr = librosa.feature.zero_crossing_rate(signal)[0]
                avg_zcr = np.mean(zcr)
                
                # Spectral centroid (brightness, higher for more intense speech)
                spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
                avg_centroid = np.mean(spectral_centroids)
                
                # Combine features into an intensity score (normalized 0-1)
                # Higher RMS + higher ZCR + higher centroid = more intense
                intensity = (avg_rms * 2 + avg_zcr * 1000 + avg_centroid / 2000) / 3
                arousal_score = min(intensity, 1.0)
            
            # Categorize arousal/intensity
            if arousal_score < 0.3:
                intensity_label = "calm"
            elif arousal_score < 0.5:
                intensity_label = "normal"
            elif arousal_score < 0.7:
                intensity_label = "elevated"
            else:
                intensity_label = "intense"
            
            results.append({
                'id': segment.id,
                'arousal_score': float(arousal_score),
                'intensity': intensity_label
            })
            
        except Exception as e:
            print(f"Error processing segment {i+1} for intensity: {e}")
            results.append({
                'id': segment.id,
                'arousal_score': 0.5,
                'intensity': "unknown"
            })
    
    print("Done classifying intensity/arousal!")
    return results

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

def describe_speech(wav_file):

    """
    If the file string starts with 'http', download audio from YouTube.
    Otherwise, assume it's a local file path.
    Returns the path to the local audio file (wav).
    """

    print("Loading Whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    print("Transcribing audio...")
    segments = transcribe_audio(model, file_path=wav_file, printing=True)
    print(f"\nFound {len(segments)} segments")
    
    # Classify intensity
    print("\nClassifying emotions for each segment (this may take a while)...")
    intensity_results = classify_intensity_and_arousal(wav_file, segments)

    print("\n" + "="*80)
    print("RESULTS: Intensity")
    print("="*80)
    
    for i, (segment, intensity) in enumerate(zip(segments, intensity_results)):
        print(f"\nSegment {segment.id}: {segment.text}")
        print(f"  Intensity: {intensity['intensity']} (arousal score: {intensity['arousal_score']:.2f})")

# Example usage:
if __name__ == "__main__":
    
    
    #############################
    file_or_link = 'https://www.youtube.com/watch?v=J7GY1Xg6X20'
    #############################

    print("Starting script...")
    print("Initializing (loading required libraries)...")

    if file_or_link.startswith('http'):
        # Download from YouTube and convert to wav
        audio_file = download_youtube_audio(file_or_link, 'data/youtube_audio.m4a')
        wav_file = 'data/youtube_audio.wav'
        mp4_to_wav(audio_file, wav_file)
    
    else:
        if file_or_link.endswith('.m4a') or file_or_link.endswith('.mp4'):
            wav_file = file_or_link.rsplit('.', 1)[0] + '.wav'
            mp4_to_wav(file_or_link, wav_file)

        else:
            raise ValueError("Unsupported file format. Please provide a .wav, .m4a, .mp4 file, or a YouTube URL.")

    describe_speech(wav_file)


