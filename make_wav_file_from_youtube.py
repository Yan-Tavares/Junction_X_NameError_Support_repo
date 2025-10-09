
import os
from pydub import AudioSegment
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel

def mp4_to_wav(mp4_path, wav_path):
    """
    Converts an MP4/M4A audio/video file to WAV format using pydub (requires ffmpeg).
    """
    
    audio = AudioSegment.from_file(mp4_path)
    audio.export(wav_path, format="wav")

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

#############################
file_or_link = 'https://www.youtube.com/watch?v=XUSiCEx3e-0'
#############################

audio_file = download_youtube_audio(file_or_link, 'data/youtube_audio.m4a')
wav_file = 'data/youtube_audio.wav'
mp4_to_wav(audio_file, wav_file)