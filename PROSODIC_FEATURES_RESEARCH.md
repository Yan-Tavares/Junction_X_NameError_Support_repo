# Research on Prosodic Features in Speech Analysis

## Overview
Prosody refers to the patterns of rhythm, stress, and intonation in speech that convey meaning beyond the literal words spoken. These features are crucial for understanding emotion, intention, and speaking style.

---

## 1. Pitch (Fundamental Frequency - F0)

### What is Pitch?
- **Definition**: Pitch is the perceived frequency of vibration of the vocal folds (vocal cords)
- **Measurement**: Fundamental frequency (F0), typically measured in Hertz (Hz)
- **Typical ranges**: 
  - Male voices: 85-180 Hz
  - Female voices: 165-255 Hz
  - Children: 250-300 Hz

### Why Pitch Represents Speech Characteristics:

#### **Emotional Expression**
- **High pitch variation**: Associated with excitement, surprise, sarcasm, or dramatic expression
- **Low pitch variation**: Associated with monotony, boredom, depression, or seriousness
- **Rising pitch**: Often indicates questions or uncertainty
- **Falling pitch**: Often indicates statements or certainty

#### **Key Research Findings**:

1. **Sarcasm Detection** (Rockwell, 2000; Attardo et al., 2003)
   - Sarcastic speech shows exaggerated pitch contours
   - Often characterized by slower speaking rate + increased pitch range
   - Reference: Attardo, S., Eisterhold, J., Hay, J., & Poggi, I. (2003). "Multimodal markers of irony and sarcasm." *Humor*, 16(2), 243-260.

2. **Emotion Recognition** (Banse & Scherer, 1996)
   - Different emotions have distinct F0 patterns:
     - **Joy**: High mean F0, high variation
     - **Anger**: High mean F0, high variation, but different contour than joy
     - **Sadness**: Low mean F0, low variation
     - **Fear**: High mean F0, moderate-high variation
   - Reference: Banse, R., & Scherer, K. R. (1996). "Acoustic profiles in vocal emotion expression." *Journal of Personality and Social Psychology*, 70(3), 614-636.

3. **Prosody and Meaning** (Cutler et al., 1997)
   - Pitch accents mark important information in utterances
   - Listeners use F0 to segment speech and identify focus
   - Reference: Cutler, A., Dahan, D., & Van Donselaar, W. (1997). "Prosody in the comprehension of spoken language: A literature review." *Language and Speech*, 40(2), 141-201.

---

## 2. Energy/Intensity (RMS - Root Mean Square)

### What is Energy/Intensity?
- **Definition**: The amplitude or "loudness" of the speech signal
- **Measurement**: RMS (Root Mean Square) energy, measured in decibels (dB)
- **Represents**: How much vocal effort and emphasis is being used

### Why Energy Represents Speech Characteristics:

#### **Key Research Findings**:

1. **Emphasis and Stress** (Fry, 1958)
   - Stressed syllables have higher intensity than unstressed ones
   - Energy peaks indicate emphasized words or concepts
   - Reference: Fry, D. B. (1958). "Experiments in the perception of stress." *Language and Speech*, 1(2), 126-152.

2. **Emotion and Energy** (Scherer, 2003)
   - **High energy**: Active emotions (anger, joy, fear)
   - **Low energy**: Passive emotions (sadness, boredom)
   - **Variable energy**: Emphatic or playful speech
   - Reference: Scherer, K. R. (2003). "Vocal communication of emotion: A review of research paradigms." *Speech Communication*, 40(1-2), 227-256.

3. **Speaking Style** (Campbell & Mokhtari, 2003)
   - Formal speech: More controlled, steady energy
   - Casual/playful speech: More energy variation
   - Reference: Campbell, N., & Mokhtari, P. (2003). "Voice quality: The 4th prosodic dimension." *Proceedings of ICPhS*, 2417-2420.

---

## 3. Speaking Rate

### What is Speaking Rate?
- **Definition**: The speed at which speech is produced
- **Measurement**: Syllables per second, phonemes per second, or Zero-Crossing Rate (ZCR) as a proxy
- **Typical range**: 4-5 syllables per second in conversational English

### Why Speaking Rate Represents Speech Characteristics:

#### **Key Research Findings**:

1. **Emotion and Rate** (Juslin & Laukka, 2003)
   - **Fast rate**: Happiness, fear, anxiety, excitement
   - **Slow rate**: Sadness, disgust, contempt, thoughtfulness
   - Reference: Juslin, P. N., & Laukka, P. (2003). "Communication of emotions in vocal expression and music performance: Different channels, same code?" *Psychological Bulletin*, 129(5), 770-814.

2. **Deception and Speaking Rate** (DePaulo et al., 2003)
   - Deceptive speech often shows hesitations and slower rate
   - Cognitive load affects speaking rate
   - Reference: DePaulo, B. M., Lindsay, J. J., Malone, B. E., Muhlenbruck, L., Charlton, K., & Cooper, H. (2003). "Cues to deception." *Psychological Bulletin*, 129(1), 74-118.

3. **Playfulness and Rate** (Bryant & Fox Tree, 2002)
   - Playful teasing shows moderate-fast rate with varied rhythm
   - Reference: Bryant, G. A., & Fox Tree, J. E. (2002). "Recognizing verbal irony in spontaneous speech." *Metaphor and Symbol*, 17(2), 99-117.

---

## 4. Spectral Features (Voice Quality)

### What are Spectral Features?
- **Spectral Centroid**: The "center of mass" of the frequency spectrum
- **Higher centroid**: Brighter, more "forward" voice quality
- **Lower centroid**: Darker, warmer, more relaxed voice quality

### Why Spectral Features Represent Speech Characteristics:

#### **Key Research Findings**:

1. **Voice Quality and Emotion** (Gobl & Chasaide, 2003)
   - Tense voice (high spectral energy): Anger, stress
   - Breathy voice (low spectral energy): Intimacy, sadness
   - Reference: Gobl, C., & Chasaide, A. N. (2003). "The role of voice quality in communicating emotion, mood and attitude." *Speech Communication*, 40(1-2), 189-212.

2. **Formant Analysis** (Laver, 1980)
   - Voice quality conveys speaker state and attitude
   - Spectral characteristics distinguish speaking styles
   - Reference: Laver, J. (1980). *The Phonetic Description of Voice Quality*. Cambridge University Press.

---

## 5. Multi-Modal Approaches to Tone Detection

### Combining Text and Prosody:

#### **Key Research**:

1. **Sarcasm Detection** (Tepperman et al., 2006)
   - Acoustic features alone: 81.6% accuracy
   - Lexical features alone: 83.3% accuracy
   - **Combined (text + prosody): 86.7% accuracy**
   - Reference: Tepperman, J., Traum, D., & Narayanan, S. (2006). "Yeah right: Sarcasm recognition for spoken dialogue systems." *Interspeech*, 1838-1841.

2. **Emotion Recognition Systems** (Schuller et al., 2011)
   - Modern systems use 6,000+ acoustic features
   - Include prosodic, spectral, and voice quality features
   - Reference: Schuller, B., Steidl, S., Batliner, A., et al. (2011). "The INTERSPEECH 2011 speaker state challenge." *Interspeech*, 3201-3204.

3. **Multimodal Sentiment Analysis** (Poria et al., 2017)
   - Combining audio, text, and visual features improves accuracy
   - Prosodic features crucial for detecting nuanced emotions
   - Reference: Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). "A review of affective computing: From unimodal analysis to multimodal fusion." *Information Fusion*, 37, 98-125.

---

## 6. Practical Applications

### What Prosodic Features Tell Us:

| Feature Pattern | Likely Interpretation |
|----------------|----------------------|
| High pitch variation + contradictory words | **Sarcasm** |
| Fast rate + high energy variation | **Excitement/Playfulness** |
| Slow rate + flat pitch | **Disbelief/Seriousness** |
| High pitch + low energy | **Fear/Uncertainty** |
| Low pitch + high energy | **Anger/Emphasis** |
| Flat pitch + steady energy | **Monotone/Boredom** |
| Rising pitch at end | **Question/Uncertainty** |
| Falling pitch at end | **Statement/Certainty** |

---

## 7. Technical Implementation References

### Audio Processing Libraries:

1. **librosa** (McFee et al., 2015)
   - Python library for audio analysis
   - Implements pitch tracking, spectral analysis, rhythm extraction
   - Reference: McFee, B., Raffel, C., Liang, D., et al. (2015). "librosa: Audio and music signal analysis in python." *Proceedings of SciPy*, 18-25.
   - Documentation: https://librosa.org/

2. **Praat** (Boersma & Weenink, 2021)
   - Gold standard for phonetic analysis
   - Detailed prosodic feature extraction
   - Reference: Boersma, P., & Weenink, D. (2021). "Praat: doing phonetics by computer."
   - Website: https://www.fon.hum.uva.nl/praat/

---

## 8. Recommended Reading

### Foundational Papers:

1. **Prosody Overview**: 
   - Cutler, A., Dahan, D., & Van Donselaar, W. (1997). "Prosody in the comprehension of spoken language." *Language and Speech*, 40(2), 141-201.

2. **Emotion in Speech**:
   - Scherer, K. R. (2003). "Vocal communication of emotion." *Speech Communication*, 40(1-2), 227-256.

3. **Sarcasm and Irony**:
   - Bryant, G. A. (2010). "Prosodic contrasts in ironic speech." *Discourse Processes*, 47(7), 545-566.

4. **Computational Approaches**:
   - Eyben, F., Wöllmer, M., & Schuller, B. (2010). "Opensmile: the munich versatile and fast open-source audio feature extractor." *ACM MM*, 1459-1462.

### Recent Surveys:

5. **Deep Learning for Speech Emotion**:
   - Akçay, M. B., & Oğuz, K. (2020). "Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers." *Speech Communication*, 116, 56-76.

---

## 9. Online Resources

1. **SUPERB Benchmark**: https://superbbenchmark.org/
   - Speech processing universal performance benchmark
   - Includes emotion recognition tasks

2. **IEMOCAP Database**: https://sail.usc.edu/iemocap/
   - Interactive Emotional Dyadic Motion Capture database
   - Gold standard for emotion recognition research

3. **Speech Prosody**: https://www.speechprosody.org/
   - International conference on speech prosody
   - Latest research and proceedings

---

## Summary

**Prosodic features are fundamental to understanding "how" something is said, not just "what" is said:**

- **Pitch (F0)**: Reveals emotional state, emphasis, questions vs. statements
- **Energy**: Shows effort, stress, and emotional activation
- **Speaking Rate**: Indicates emotional arousal and cognitive load
- **Spectral Quality**: Conveys voice tension, breathiness, and speaker state

**For detecting tone** (sarcasm, playfulness, disbelief):
- You need to analyze **patterns** and **combinations** of features
- Text + prosody together work better than either alone
- Context is crucial - same prosodic pattern can mean different things in different contexts

The features we extracted in your code are based on decades of speech research showing these acoustic properties reliably correlate with emotional and intentional states in speech.
