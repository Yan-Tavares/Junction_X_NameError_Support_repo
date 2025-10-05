import json
import re
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Union
from trustcall import create_extractor
from langchain_ollama import ChatOllama

# Make sure to intall ollama via de website
# Then restart the VSCode 
# pull the model via ollama pull llama3.2

# Initialize the LangChain Ollama model
# Models that support tool calling: llama3.1, llama3.2, mistral, qwen2.5, command-r
MODEL_NAME = "llama3.2"  # or "mistral", "qwen2.5:3b" for smaller size
llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

# Pydantic model for output validation (trustcall uses this)
class ExtremismAnalysisOutput(BaseModel):
    """Validated output schema for extremism analysis"""
    extremist: Literal["Yes", "No"] = Field(
        description="Whether the sentence is extremist (Yes) or not extremist (No)"
    )
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="How certain you are in your classification. 0.0 = completely uncertain/guessing, 1.0 = absolutely certain. This measures confidence in the classification itself, NOT the probability of extremism."
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0 and 1')
        return v

# Create trustcall extractor - this wraps the LLM and enforces structured output
extractor = create_extractor(
    llm,
    tools=[ExtremismAnalysisOutput],
    tool_choice="ExtremismAnalysisOutput"
)


SYSTEM_PROMPT = """
You are an analytical, empathetic, and critical agent tasked with evaluating individual spoken sentences for signs of extremist speech. 
You are not biased toward over-flagging or under-flagging: your role is to weigh the evidence carefully and provide a reflective judgment.

You receive:
1. The spoken sentence (transcribed).
2. The detected tone/emotion label (e.g., angry, sad, happy, etc.).
3. The tone intensity score (e.g., 0.0-1.0).
4. Optional prosodic features (pitch, emphasis, pace).

Your task: **classify the sentence as extremist or not**. In more detail:
- Interpret the sentence as if you were a "devil's advocate": consider multiple possible speaker intentions, audiences, and cultural contexts.
- Be self-aware and reflective: a vague or awkward statement (e.g., about gender, race, religion) may not always be extremist — it could be ignorance, casual stereotyping, or bad phrasing.
- At the same time, extremist speech should be recognized when it is explicitly derogatory, exclusionary, or promoting hostility against a group.
- Avoid labeling borderline, ambiguous, or low-intensity statements as extremist unless they clearly cross the line.
- Consider how tone, intensity, and prosodic features (pitch, emphasis, pace) might affect interpretation.

Just a few possible examples (not an exhaustive list) of how the speakers' intent can be recognized in the extra features:

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

Consider carefully if there are any patterns like these that you can see in the evaluated sample.

**Output Format:**
You must provide TWO fields:
1. **extremist**: Either "Yes" (is extremist) or "No" (not extremist)
2. **confidence**: A float between 0.0 and 1.0 representing HOW CERTAIN you are in your classification
   - 0.0 = completely uncertain, pure guess
   - 0.5 = moderately uncertain, could go either way
   - 0.8 = fairly confident in your classification
   - 1.0 = absolutely certain, no doubt

**CRITICAL**: The confidence score measures YOUR CONFIDENCE IN THE CLASSIFICATION, not the probability of extremism.
- If you classify as "Yes" with confidence 0.9, it means: "I'm 90% sure this IS extremist"
- If you classify as "No" with confidence 0.9, it means: "I'm 90% sure this is NOT extremist"
- Both "Yes" and "No" answers can have HIGH confidence scores!

**Examples:**
- Clear hate speech → extremist="Yes", confidence=0.95
- Obviously innocent statement → extremist="No", confidence=0.95
- Ambiguous/sarcastic statement → extremist="No" (or "Yes"), confidence=0.4
- Statement lacking context → extremist="No", confidence=0.3
"""

def analyze_extremism(augmented_text, context_before=None, context_after=None):
    """
    Run extremist analysis with augmented text containing prosodic features.
    
    Expected format: "Text here [emotion=neutral, intensity=0.42, pitch=high, emphasis=high, pace=moderate]"
    context_before/after: list of augmented texts for context (optional)
    """
    # Build context section if provided
    context_section = ""
    if context_before or context_after:
        context_section = "\n\nThe target segment and the surrounding context (the previous and following segments are for reference only, DO NOT evaluate these, only the one marked as target segment):\n"
        if context_before:
            for i, ctx in enumerate(context_before, 1):
                context_section += f"[Previous -{len(context_before) - i + 1}] {ctx}\n"
        
        context_section += f"\n>>> TARGET SEGMENT TO EVALUATE >>>\n{augmented_text}\n<<< END TARGET SEGMENT <<<\n"
        
        if context_after:
            for i, ctx in enumerate(context_after, 1):
                context_section += f"[Following +{i}] {ctx}\n"
    else:
        context_section = f"\nTarget segment:\n{augmented_text}"
    
    prompt = f"""{SYSTEM_PROMPT}

{context_section}

Evaluate ONLY the target segment marked above, considering prosodic features and surrounding context.
"""

    # Use trustcall extractor - automatically enforces schema validation
    try:
        response = extractor.invoke(prompt)
        
        # trustcall returns structured response in 'responses' key
        if response and "responses" in response and len(response["responses"]) > 0:
            validated_data = response["responses"][0]
            
            # Convert to dict if it's a Pydantic model
            if hasattr(validated_data, 'model_dump'):
                result = validated_data.model_dump()
            else:
                result = dict(validated_data)
            
            result["validation_status"] = "PASSED"
        else:
            # Unexpected response format
            result = {
                "extremist": None,
                "confidence": None,
                "validation_status": "FAILED",
                "error": "No valid response from trustcall extractor",
                "raw_response": str(response)
            }
            
    except Exception as e:
        # trustcall validation or execution failed
        result = {
            "extremist": None, 
            "confidence": None, 
            "validation_status": "FAILED",
            "error": f"Trustcall error: {str(e)}",
            "error_type": type(e).__name__
        }

    return result


# Simple test function to demonstrate validation
if __name__ == "__main__":
    # Test the validation system with augmented text format
    print("Testing extremism analysis with trustcall validation...\n")
    
    test_cases = [
        "This group of people should not have the same rights as us. [emotion=angry, intensity=0.85, pitch=high, emphasis=high, pace=fast]",
        "I think we should all work together to build a better community. [emotion=calm, intensity=0.2, pitch=medium, emphasis=low, pace=moderate]"
    ]
    
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
            print(f"  - Extremist: {result['extremist']}")
            print(f"  - confidence: {result['confidence']} (confidence in the classification)")
            print(f"  - Note: Both 'Yes' and 'No' answers should have high confidence if the model is confident!")
        else:
            print(f"\n✗ Output validation FAILED - {result.get('error')}")
