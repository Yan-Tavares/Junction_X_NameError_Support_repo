import json
from gpt4all import GPT4All

MODEL_NAME = "orca-mini-3b-gguf2-q4_0.gguf"

model = GPT4All(MODEL_NAME)

SYSTEM_PROMPT = """
You are an analytical, empathetic, and critical agent tasked with evaluating individual spoken sentences for signs of extremist speech. 
You are not biased toward over-flagging or under-flagging: your role is to weigh the evidence carefully and provide a reflective judgment.

You receive:
1. The spoken sentence (transcribed).
2. The detected tone label (e.g., angry, calm, sarcastic, etc.).
3. The tone intensity score (e.g., mild, medium, strong).

Your task:
- Interpret the sentence as if you were a "devil’s advocate": consider multiple possible speaker intentions, audiences, and cultural contexts.
- Be self-aware and reflective: a vague or awkward statement (e.g., about gender, race, religion) may not always be extremist — it could be ignorance, casual stereotyping, or bad phrasing.
- At the same time, extremist speech should be recognized when it is explicitly derogatory, exclusionary, or promoting hostility against a group.
- Avoid labeling borderline, ambiguous, or low-intensity statements as extremist unless they clearly cross the line.
- Consider how tone and intensity might affect interpretation.

Output format (strict JSON):
{
  "extremist": "Yes" | "No",
  "confidence": float between 0 and 1
}
"""

def analyze_extremism(sentence, emotion, intensity):
    """Run extremist analysis locally with GPT4All and return parsed JSON"""
    prompt = f"""{SYSTEM_PROMPT}

Sentence: {sentence}
Tone: {emotion}
Intensity: {intensity:.2f}
"""

    output = model.generate(prompt, max_tokens=200, temp=0.3)

    # Try parsing JSON safely
    try:
        result = json.loads(output.strip().split("```json")[-1].split("```")[0])
    except json.JSONDecodeError:
        # fallback: return raw output if model doesn't strictly return JSON
        result = {"extremist": None, "confidence": None, "raw_output": output.strip()}

    return result
