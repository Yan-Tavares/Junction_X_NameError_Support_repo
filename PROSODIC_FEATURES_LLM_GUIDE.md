# Prosodic Features - Quick Reference for LLM Context

## Feature Interpretation Guide

When feeding augmented text to an LLM, provide this context:

### Prosodic Feature Categories:

#### **pitch** (pitch_expressiveness)
- `high`: Highly expressive (sarcasm, excitement, dramatization)
- `medium`: Moderate/conversational
- `low`: Flat/monotone (serious, disinterested, boredom)

#### **emphasis**
- `high`: Varying intensity (emphatic speech, playfulness)
- `medium`: Moderate variation
- `low`: Steady (calm, subdued)

#### **pace** (speaking rate)
- `fast`: Fast speech (excited, nervous, playful)
- `moderate`: Normal conversational pace
- `slow`: Slow speech (serious, thoughtful, skeptical, disbelief)

#### **voice_quality** (optional in augmented text)
- `bright`: Energetic, tense voice
- `neutral`: Normal voice quality
- `warm`: Relaxed, intimate voice

---

## Example Augmented Text

```
"I'm totally fine with that decision. [emotion=neutral, intensity=0.42, pitch=high, emphasis=high, pace=moderate]"
```

### LLM Prompt Example:

```
Analyze the following speech segments for tone (sarcasm, playfulness, disbelief, etc.).

Context for prosodic features:
- pitch: high=expressive/dramatic, medium=conversational, low=flat/monotone
- emphasis: high=emphatic/playful, medium=moderate, low=steady/calm
- pace: fast=excited/playful, moderate=normal, slow=serious/skeptical
- emotion: detected emotion category
- intensity: emotion confidence (0-1)

Segments:
1. "I'm totally fine with that decision. [emotion=neutral, intensity=0.42, pitch=high, emphasis=high, pace=moderate]"
   → Likely sarcastic (high pitch + high emphasis + contradictory neutral emotion)

2. "That's absolutely wonderful news! [emotion=happy, intensity=0.85, pitch=high, emphasis=high, pace=fast]"
   → Likely genuine excitement (aligned emotion and prosody)

3. "Sure, whatever you say. [emotion=neutral, intensity=0.38, pitch=high, emphasis=high, pace=slow]"
   → Likely sarcastic or dismissive (high pitch + slow pace + neutral words)

4. "I need to think about this carefully. [emotion=neutral, intensity=0.65, pitch=low, emphasis=low, pace=slow]"
   → Likely serious/thoughtful (all features align)

Classify tone for each segment.
```

---

## Common Patterns

| Tone | Typical Pattern |
|------|----------------|
| **Sarcasm** | pitch=high + emphasis=high + contradictory words + emotion=neutral |
| **Playfulness** | pitch=medium/high + emphasis=high + pace=fast/moderate |
| **Disbelief** | pitch=high OR low + pace=slow + questioning words |
| **Serious** | pitch=low + emphasis=low + pace=moderate/slow |
| **Excitement** | pitch=high + emphasis=high + pace=fast + emotion=happy |
| **Boredom** | pitch=low + emphasis=low + pace=slow + emotion=neutral |
| **Anger** | pitch=high + emphasis=high + pace=fast + emotion=angry |
| **Thoughtful** | pitch=low + emphasis=low/medium + pace=slow |

---

## Feature Mapping (Technical Details)

### Numerical Thresholds Used:

**pitch_expressiveness** (pitch_variation coefficient):
- high: > 0.3
- medium: 0.15 - 0.3
- low: < 0.15

**emphasis** (energy_variation coefficient):
- high: > 0.5
- medium: 0.3 - 0.5
- low: < 0.3

**pace** (zero-crossing rate):
- fast: > 0.15
- moderate: 0.08 - 0.15
- slow: < 0.08

**voice_quality** (spectral centroid in Hz):
- bright: > 2000 Hz
- neutral: 1000 - 2000 Hz
- warm: < 1000 Hz
