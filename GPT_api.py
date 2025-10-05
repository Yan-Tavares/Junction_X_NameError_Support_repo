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
3. The emotion confidence score (e.g., 0.0-1.0).
4. Prosodic features (pitch, emphasis, pace).

Your task: **Provide THREE probability scores** that sum to 1.0:
- **p_safe**: Probability this is safe/normal content (0.0-1.0)
- **p_uncertain**: Probability this is uncertain/offensive but not clearly extremist (0.0-1.0)  
- **p_extremist**: Probability this is extremist content (0.0-1.0)

Guidelines:
- Interpret the sentence considering multiple possible speaker intentions, audiences, and cultural contexts.
- Be self-aware and reflective: vague or awkward statements may be ignorance, casual stereotyping, or bad phrasing (→ uncertain).
- Extremist speech should be recognized when it is explicitly derogatory, exclusionary, or promoting hostility against a group.
- Consider how tone, intensity, and prosodic features affect interpretation.

**Prosodic Feature Patterns:**

| Tone | Typical Pattern | Classification Hint |
|------|----------------|-------------------|
| **Sarcasm** | pitch=high + emphasis=high + contradictory words + emotion=neutral | Often safer than literal interpretation |
| **Playfulness** | pitch=medium/high + emphasis=high + pace=fast | Usually safe unless content is clearly hostile |
| **Disbelief** | pitch=high OR low + pace=slow + questioning | Often uncertain, rarely extremist |
| **Serious** | pitch=low + emphasis=low + pace=moderate/slow | Take literal meaning more seriously |
| **Anger** | pitch=high + emphasis=high + pace=fast + emotion=angry | Higher chance of extremism if targeting groups |
| **Thoughtful** | pitch=low + emphasis=low/medium + pace=slow | Usually safe or uncertain |

**Decision Process:**

1. **High p_safe (0.7-1.0)**: 
   - Clearly non-hostile content
   - Neutral or positive statements
   - Sarcasm/jokes that are obviously not serious
   
2. **High p_uncertain (0.5-0.8)**:
   - Ambiguous statements that could be interpreted multiple ways
   - Casual stereotypes or insensitive comments
   - Content that's offensive but not promoting violence/hatred
   - Borderline cases where context is unclear
   
3. **High p_extremist (0.5-1.0)**:
   - Explicit derogatory language toward groups
   - Calls for exclusion, harm, or denial of rights
   - Promoting hostility based on identity (race, religion, gender, etc.)
   - Clear hate speech

**CRITICAL**: The three probabilities MUST sum to 1.0 (or very close, like 0.99-1.01).

**Examples:**

Example 1: "I think everyone deserves equal rights. [emotion=calm, emotion confidence=0.8, pitch=medium, emphasis=low, pace=moderate]"
→ p_safe=0.95, p_uncertain=0.05, p_extremist=0.0

Example 2: "Those people are a bit weird, honestly. [emotion=neutral, emotion confidence=0.6, pitch=high, emphasis=high, pace=fast]"
→ p_safe=0.3, p_uncertain=0.65, p_extremist=0.05

Example 3: "They should be removed from our country. [emotion=angry, emotion confidence=0.9, pitch=high, emphasis=high, pace=moderate]"
→ p_safe=0.05, p_uncertain=0.15, p_extremist=0.8

Example 4: "Oh sure, they're TOTALLY gonna save the world... [emotion=neutral, emotion confidence=0.7, pitch=high, emphasis=high, pace=moderate]"
(Sarcastic tone detected → likely not serious threat)
→ p_safe=0.7, p_uncertain=0.25, p_extremist=0.05
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
        "This group of people should not have the same rights as us. [emotion=angry, emotion confidence=0.85, pitch=high, emphasis=high, pace=fast]",
        "I think we should all work together to build a better community. [emotion=calm, emotion confidence=0.8, pitch=medium, emphasis=low, pace=moderate]",
        "Those people are a bit weird, honestly. [emotion=neutral, emotion confidence=0.6, pitch=high, emphasis=high, pace=fast]"
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
            print(f"  - p_safe: {result['p_safe']:.3f}")
            print(f"  - p_uncertain: {result['p_uncertain']:.3f}")
            print(f"  - p_extremist: {result['p_extremist']:.3f}")
            total = result['p_safe'] + result['p_uncertain'] + result['p_extremist']
            print(f"  - Sum: {total:.3f} (should be ~1.0)")
        else:
            print(f"\n✗ Output validation FAILED - {result.get('error')}")
