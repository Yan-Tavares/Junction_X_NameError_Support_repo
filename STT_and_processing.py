# ffmpeg required, install via choco
# sympy-1.13.1
# torch = torch-2.5.1+cu121
# torchaudio-2.5.1+cu121


# Use faster-whisper for word-level timestamps
from faster_whisper import WhisperModel

# Load the model ("base" can be changed to "small", "medium", etc.)
model = WhisperModel("base", device="cpu", compute_type="int8")

# Transcribe audio file with word-level timestamps
segments, info = model.transcribe("data/first_hateful_speech.m4a", word_timestamps=True)

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

print(f"Detected language: {info.language}\n")
print("Transcription with word-level timestamps:")

for segment in segments:
	print(f"[Segment {segment.id}] {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
	if segment.words:
		for word in segment.words:
			print(f"    {word.word} ({word.start:.2f}s - {word.end:.2f}s)")

