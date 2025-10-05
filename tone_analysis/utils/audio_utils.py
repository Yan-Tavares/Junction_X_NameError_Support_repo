import librosa
import soundfile as sf
from pydub import AudioSegment

def to_wav(input_path, output_path="output.wav", sr=16000):
    """
    Convert mp3/mp4/wav to clean 16kHz mono WAV.
    Returns path to the new WAV file.
    """
    if input_path.endswith(".wav"):
        # Load and resave to enforce correct format
        y, _ = librosa.load(input_path, sr=sr, mono=True)
        sf.write(output_path, y, sr)
    elif input_path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(input_path)
        sound = sound.set_frame_rate(sr).set_channels(1)
        sound.export(output_path, format="wav")
    elif input_path.endswith(".mp4"):
        sound = AudioSegment.from_file(input_path, "mp4")
        sound = sound.set_frame_rate(sr).set_channels(1)
        sound.export(output_path, format="wav")
    else:
        raise ValueError("Unsupported file type. Use .wav, .mp3, or .mp4")
    return output_path

def chunk_audio(wav_path, chunk_size_s=10):
    """
    Split WAV into chunks of `chunk_size_s` seconds.
    Returns list of (chunk_file, start_time, end_time).
    """
    y, sr = librosa.load(wav_path, sr=None)
    chunk_len = chunk_size_s * sr
    chunks = []
    for i in range(0, len(y), chunk_len):
        chunk = y[i:i+chunk_len]
        start, end = i/sr, min((i+chunk_len)/sr, len(y)/sr)
        out_file = f"{wav_path}_chunk_{int(start)}.wav"
        sf.write(out_file, chunk, sr)
        chunks.append((out_file, start, end))
    return chunks
