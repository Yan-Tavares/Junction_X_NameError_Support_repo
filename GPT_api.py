import json
import re
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Union
from langchain_ollama import ChatOllama

########### Instructions to be able to use Ollama ####################
# Make sure to intall ollama via the website
# Then restart the VSCode 
# pull the model via > ollama pull llama3.2:1b or > ollama pull llama3.2:3b
###################


# Initialize the LangChain Ollama model
# Models that support tool calling: llama3.1, llama3.2, mistral, qwen2.5, command-r
MODEL_NAME = "llama3.1:8b"  # or "mistral", "qwen2.5:3b" for smaller size

print(f"Loading LLM model: {MODEL_NAME}...")
llm = ChatOllama(
    model=MODEL_NAME, 
    temperature=0,
    seed=42,
    num_gpu=1 
)
print(f"LLM model loaded successfully.")

# Pydantic model for output validation (trustcall uses this)
class ExtremismAnalysisOutput(BaseModel):
    """Validated output schema for extremism analysis with three probability scores"""

    p_safe: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Probability that the content is safe/normal (0.0 to 1.0)"
    )
    p_uncertain: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Probability that the content is uncertain/offensive but not clearly extremist (0.0 to 1.0)"
    )
    p_extremist: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Probability that the content is extremist (0.0 to 1.0)"
    )
    
    @field_validator('p_safe', 'p_uncertain', 'p_extremist')
    @classmethod
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Probability must be between 0 and 1')
        return v
    
    def model_post_init(self, __context):
        """Validate that probabilities sum to approximately 1.0"""
        total = self.p_safe + self.p_uncertain + self.p_extremist
        if not (0.95 <= total <= 1.05):  # Allow small floating point errors
            raise ValueError(f'Probabilities must sum to ~1.0, got {total}')

# Use LangChain's native structured output instead of trustcall (more reliable)
extractor = llm.with_structured_output(ExtremismAnalysisOutput)


PROMPT_FOR_CONTEXT_ANALYSIS = """
You are an analytical, and critical agent.\n
You will receive a transcribed text from a spoken audio.\n
You must read the provided text carefully and summarize it in the maximum of 3 sentences.\n
Your task is to provide a context to another agent who will analyze individual sentences for extremist speech.\n
So you must give enough context that clarify if the entire text seems to promot extremist views or not.\n 

** CRITICAL**:\n
- Read it literally, do not try to assume hidden meanings.\n
- ONLY write the summary, nothing else. \n
- Do NOT introduce you or the summary. \n
- Do NOT ask for further clarifications.\n
- Do NOT give your opinion about if it is extremist or not, focus on the content only.\n
- Keep the summary very brief (max 3 sentences)\n
"""

PROMPT_FOR_EXTREMISM_ANALYSIS = """
You are an analytical, empathetic, and critical agent tasked with evaluating individual spoken sentences for signs of extremist speech. 
You are not biased toward over-flagging or under-flagging: your role is to weigh the evidence carefully and provide a reflective judgment.
\n
You receive:\n
1. Short summary of the entire audio.\n
2. The spoken sentence (transcribed).\n
3. The detected tone/emotion label (e.g., angry, sad, happy, etc.).\n
4. The emotion confidence score (e.g., 0.0-1.0).\n
5. Prosodic features (pitch, emphasis, pace).\n
\n
**Definition of extremism**:\n
A speech is extremist if it explicitly or implicitly falls into one or more of these categories:\n
- 1 Negate or destroy the fundamental rights and freedoms of others; or \n
- 2 undermine, overturn or replace democratic rights\n
- 3 intentionally create a permissive environment for others to achieve the results in (1) or (2).\n
- 4 distributes disinformation. (e.g, claims that are not common sense and are not supported by evidence)\n
\n
**Your task:**\n
**Provide THREE probability scores** for the TARGET SENTENCE that sum to 1.0:\n
- **p_normal**: Probability this is safe/normal content (0.0-1.0).\n
- **p_uncertain**: Probability this is uncertain/offensive but not clearly extremist (0.0-1.0).\n 
- **p_extremist**: Probability this is extremist content (0.0-1.0).\n
\n
Guidelines:\n
- Use the context provided to understand the speaker perspective, but **FOCUS ON THE TARGET SENTENCE**\n
- Even when the context is extremist, the target sentence might be safe/normal\n
- Be self-aware and reflective: vague or awkward statements may be ignorance, casual stereotyping, or bad phrasing (→ uncertain).\n
- Consider how emotion and prosodic features affect interpretation.\n
\n

**Decision Process:**
\n
1. **High p_normal (0.7-1.0)**: 
   - Clearly non-hostile content
   - Neutral or positive statements
   - Sarcasm/jokes that are obviously not serious
\n
2. **High p_offensive (0.5-0.8)**:
   - Ambiguous statements that could be interpreted multiple ways
   - Casual stereotypes or insensitive comments
   - Content that's offensive but not promoting violence/hatred
   - Borderline cases where context is unclear
\n 
3. **High p_extremist (0.5-1.0)**:
   - Explicit derogatory language toward groups
   - Calls for exclusion, harm, or denial of rights
   - Promoting hostility based on identity (race, religion, gender, etc.)
   - Clear hate speech
\n
**CRITICAL**: \n
- The three probabilities MUST sum to 1.0 (or very close, like 0.99-1.01).\n
- Only answer in the following format (no explanations, no extra text):\n
- - p_safe=0.XX, p_uncertain=0.XX, p_extremist=0.XX\n

**Examples:**

Example 1:
SHORT SUMMARY OF ENTIRE AUDIO: \n
The speaker addresses the problems regarding inequality. He points out how racism and discrimination can severely harm an individual's sense of belonging and self-worth, ultimately impacting their ability to advocate for equality and human rights by stating that.\n
TARGET SENTENCE:\n
START >>I think everyone deserves equal rights [emotion=sad, emotion confidence=0.74, pitch=low, emphasis=low/medium, pace=slow] <<END \n
OUTPUT:\n 
p_safe=0.95, p_uncertain=0.05, p_extremist=0.0\n

"""
# Example 2:
# SHORT SUMMARY OF ENTIRE AUDIO:
# The speaker discusses the challenges of integrating different cultures and expresses frustration about recent changes in the neighborhood, mentioning concerns about safety and traditions being lost.
# SENTENCE TO ANALYZE THE EXTREMISM:
# START >>These people are ruining our community and should go back to where they came from [emotion=anger, emotion confidence=0.74, pitch=high, emphasis=high, pace=fast] << END
# OUTPUT:\n
# p_safe=0.05, p_uncertain=0.15, p_extremist=0.8

# Example 3:
# SHORT SUMMARY OF ENTIRE AUDIO:
# The speaker jokes about stereotypes in movies, using exaggerated voices and making it clear they are not being serious, and then comments on how media can influence perceptions.
# SENTENCE TO ANALYZE THE EXTREMISM:
# START >>Oh sure, all villains have accents, right? [emotion=neutral, emotion confidence=0.68, pitch=high, emphasis=high, pace=moderate]<< END
# OUTPUT:\n
# p_safe=0.7, p_uncertain=0.25, p_extremist=0.05


#
# **Prosodic Feature Patterns:**
# |------|----------------|
# | **Sarcasm** | emotion=neutral + pitch=high + emphasis=high + contradictory words|
# | **Playfulness** | emotion=happy OR anger + pitch=medium/high + emphasis=high + pace=fast/moderate |
# | **Disbelief** | pitch=high OR low + pace=slow + questioning words |
# | **Serious** | emotion=neutral + pitch=low + emphasis=low + pace=moderate/slow |
# | **Excitement** | emotion=happy + pitch=high + emphasis=high + pace=fast +  |
# | **Boredom** | emotion=neutral + pitch=low + emphasis=low + pace=slow +  |
# | **Anger** | emotion=angry + pitch=high + emphasis=high + pace=fast  |
# | **Thoughtful** | emotion=neutral + pitch=low + emphasis=low/medium + pace=slow |

def summarize_text(segments):
    # Build context section if provided
    full_text = " ".join([segment.text for segment in segments])
    print(full_text)
    context_prompt = f"""
    {PROMPT_FOR_CONTEXT_ANALYSIS}
    Text to summarize:
    {full_text}
    """
    try:
        context_summary = (llm.invoke(context_prompt)).content
    except Exception as e:
        context_summary = f"Failed to generate summary: {str(e)}"

    return context_summary
    
def analyze_extremism(segments, context_summary):
    """
    Run extremist analysis with augmented text containing prosodic features.
    Analyzes all segments and returns a list of results.
    
    Example format: "Text here [emotion=neutral, intensity=0.42, pitch=high, emphasis=high, pace=moderate]"
    """
    
    results = []
    
    for segment in segments:
        # Get the augmented text from the segment
        augmented_text = segment.augmented_text
        
        prompt = f"""{PROMPT_FOR_EXTREMISM_ANALYSIS}
        SHORT SUMMARY OF ENTIRE AUDIO:\n
        {context_summary}\n
        SENTENCE TO ANALYZE THE EXTREMISM:\n
        START >> {augmented_text} << END
        OUTPUT:
        """

        # Use LangChain's structured output - automatically enforces schema validation
        try:
            response = extractor.invoke(prompt)
            
            # LangChain's with_structured_output returns the Pydantic model directly
            if hasattr(response, 'model_dump'):
                result = response.model_dump()
            else:
                result = dict(response)
            
            result["validation_status"] = "PASSED"
                
        except Exception as e:
            # Validation or execution failed
            result = {
                "p_safe": 0.7,
                "p_uncertain": 0.2,
                "p_extremist": 0.1,
                "validation_status": "FAILED",
                "error": f"LLM error: {str(e)}",
                "error_type": type(e).__name__
            }
        
        results.append(result)
    
    return results


# Simple test function to demonstrate validation
if __name__ == "__main__":

    audio_path = "src/testing/test_extremist_audio.wav"


    print("Testing extremism analysis with trustcall validation...\n")

    
    

    for i, test_augmented_text in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        
        result = analyze_extremism(test_augmented_text)
        
        print(f"Input (augmented text):\n{test_augmented_text}")
        print(f"\nOutput:")
        print(json.dumps(result, indent=2))
        
        # Check validation status
        if result.get("validation_status") == "PASSED":
            print("\n✓ Output validation PASSED - Model returned correct format")
            print(f"  - p_safe: {result['p_safe']:.3f}")
            print(f"  - p_uncertain: {result['p_uncertain']:.3f}")
            print(f"  - p_extremist: {result['p_extremist']:.3f}")
            total = result['p_safe'] + result['p_uncertain'] + result['p_extremist']
            print(f"  - Sum: {total:.3f} (should be ~1.0)")
        else:
            print(f"\n✗ Output validation FAILED - {result.get('error')}")
