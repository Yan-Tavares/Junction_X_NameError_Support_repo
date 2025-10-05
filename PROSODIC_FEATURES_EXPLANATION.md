# Prosodic Feature Extraction for Tone Analysis

## Overview
We've added prosodic feature extraction to help detect speaking tone and style (sarcasm, playfulness, disbelief, etc.) in audio segments.

## Features Extracted

### 1. **Pitch Features** (F0 Analysis)
- **Mean Pitch**: Average fundamental frequency
- **Pitch Standard Deviation**: How much pitch varies
- **Pitch Range**: Difference between highest and lowest pitch
- **Pitch Variation Coefficient**: Normalized measure of pitch expressiveness

**What it indicates:**
- High variation → Expressive, dramatic, possibly sarcastic
- Moderate variation → Normal conversational
- Low variation → Monotone, serious, disinterested

### 2. **Energy/Intensity Features**
- **Mean Energy**: Average volume/loudness
- **Energy Standard Deviation**: Variation in volume
- **Energy Variation Coefficient**: Normalized measure of emphasis

**What it indicates:**
- High variation → Emphatic speech, playfulness
- Moderate variation → Normal speech
- Low variation → Calm, subdued, or monotone

### 3. **Speaking Rate Proxy**
- Uses Zero Crossing Rate as an indicator of speech tempo

**What it indicates:**
- High rate → Excited, nervous, playful
- Low rate → Serious, thoughtful, skeptical, disbelief
- Moderate → Normal conversation

### 4. **Spectral Features**
- **Spectral Centroid**: "Brightness" of the voice

**What it indicates:**
- Bright voice → Higher energy, possibly excited or tense
- Darker voice → Warmer, calmer, more relaxed

## How It Works

1. **Extract Features**: For each audio segment, we extract all prosodic features
2. **Generate Description**: Convert numerical features into human-readable tone descriptions
3. **Augment Text**: Add the tone description to the transcribed text

## Example Output

```
Segment 1: I'm totally fine with that decision.
Time: 0.00s - 3.45s
--------------------------------------------------------------------------------
Emotion: neutral
Intensity: 0.42
Tone Analysis: highly expressive pitch (may indicate sarcasm, excitement, or dramatization); 
               varying intensity (emphatic speech, possible playfulness); 
               moderate speaking rate
--------------------------------------------------------------------------------
Augmented Text:
I'm totally fine with that decision. [Emotion: neutral, Intensity: 0.42, 
Tone: highly expressive pitch (may indicate sarcasm, excitement, or dramatization); 
varying intensity (emphatic speech, possible playfulness); moderate speaking rate]
```

## Usage for LLM Context

The augmented text can be fed to an LLM for:
- Detecting sarcasm (high pitch variation + contradictory words)
- Identifying playfulness (varying energy + moderate-high speaking rate)
- Spotting disbelief (flat pitch or exaggerated pitch + questioning words)
- Understanding emphasis and emotional nuance

## Interpreting Combinations

### Sarcasm Indicators:
- Highly expressive pitch + neutral/positive words + contradictory context
- Example: "Oh great, just what I needed" with exaggerated pitch

### Playfulness:
- Moderate-high pitch variation + varying intensity + faster rate
- Bright voice quality

### Disbelief/Skepticism:
- Slower speaking rate + questioning tone + flat or exaggerated pitch
- Example: "Really? You expect me to believe that?"

### Serious/Formal:
- Flat pitch + steady intensity + moderate rate
- Darker voice quality

## Next Steps

You can:
1. Feed the augmented text to an LLM (GPT-4, Claude, etc.) for final tone classification
2. Use the numerical features to train a custom classifier
3. Set thresholds to automatically flag potential sarcasm/playfulness
4. Combine with sentiment analysis for more accurate results
