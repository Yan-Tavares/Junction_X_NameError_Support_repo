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
        description="Whether the sentence is extremist"
    )
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
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

Your task:
- Interpret the sentence as if you were a "devil's advocate": consider multiple possible speaker intentions, audiences, and cultural contexts.
- Be self-aware and reflective: a vague or awkward statement (e.g., about gender, race, religion) may not always be extremist — it could be ignorance, casual stereotyping, or bad phrasing.
- At the same time, extremist speech should be recognized when it is explicitly derogatory, exclusionary, or promoting hostility against a group.
- Avoid labeling borderline, ambiguous, or low-intensity statements as extremist unless they clearly cross the line.
- Consider how tone, intensity, and prosodic features (pitch, emphasis, pace) might affect interpretation.

Output format (strict JSON):
{
  "extremist": "Yes" | "No",
  "confidence": float between 0 and 1
}
"""

def analyze_extremism(augmented_text):
    """
    Run extremist analysis with augmented text containing prosodic features.
    
    Expected format: "Text here [emotion=neutral, intensity=0.42, pitch=high, emphasis=high, pace=moderate]"
    """
    # The augmented text contains everything - just pass it to the LLM
    prompt = f"""{SYSTEM_PROMPT}

Augmented sentence with prosodic features:
{augmented_text}

Analyze this sentence considering all the prosodic features provided in the brackets.
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
    
    test_augmented_text = "This group of people should not have the same rights as us. [emotion=angry, intensity=0.85, pitch=high, emphasis=high, pace=fast]"
    
    result = analyze_extremism(test_augmented_text)
    
    print(f"Input (augmented text):\n{test_augmented_text}")
    print(f"\nOutput:")
    print(json.dumps(result, indent=2))
    
    # Check validation status
    if result.get("validation_status") == "PASSED":
        print("\n✓ Output validation PASSED - Model returned correct format")
        print(f"  - Extremist: {result['extremist']}")
        print(f"  - Confidence: {result['confidence']}")
    else:
        print(f"\n✗ Output validation FAILED - {result.get('error')}")
