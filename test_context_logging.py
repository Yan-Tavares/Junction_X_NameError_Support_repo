import json
import sys
from GPT_api import analyze_extremism, SYSTEM_PROMPT

def test_with_detailed_logging(json_path, num_segments=5):
    """
    Test the LLM with detailed logging to see exactly what it receives.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        augmented_texts_dict = json.load(f)

    # Convert to list and take only first num_segments
    segment_items = list(augmented_texts_dict.items())[:num_segments]
    
    print("=" * 100)
    print(f"TESTING WITH {len(segment_items)} SEGMENTS")
    print("=" * 100)
    print()
    
    # Process each segment with context
    for idx, (segment_id, augmented_text) in enumerate(segment_items):
        print("\n" + "=" * 100)
        print(f"PROCESSING SEGMENT {segment_id} (index {idx})")
        print("=" * 100)
        
        # Extract context: up to 2 previous and 2 following segments
        context_before = [segment_items[i][1] for i in range(max(0, idx-2), idx)]
        context_after = [segment_items[i][1] for i in range(idx+1, min(len(segment_items), idx+3))]
        
        print(f"\n📍 Current segment text:")
        print(f"   {augmented_text}")
        print(f"\n📍 Context before: {len(context_before)} segments")
        for i, ctx in enumerate(context_before):
            print(f"   [{i+1}] {ctx[:100]}...")
        print(f"\n📍 Context after: {len(context_after)} segments")
        for i, ctx in enumerate(context_after):
            print(f"   [{i+1}] {ctx[:100]}...")
        
        # Build the exact prompt that will be sent to LLM
        print("\n" + "🔥" * 50)
        print("FULL PROMPT SENT TO LLM:")
        print("🔥" * 50)
        
        # Build context section (same logic as in GPT_api.py)
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
        
        full_prompt = f"""{SYSTEM_PROMPT}

{context_section}

Evaluate ONLY the target segment marked above, considering prosodic features and surrounding context.
"""
        
        print(full_prompt)
        print("\n" + "🔥" * 50)
        print("END OF PROMPT")
        print("🔥" * 50)
        
        # Now actually call the LLM
        print("\n⏳ Calling LLM...")
        llm_result = analyze_extremism(
            augmented_text, 
            context_before=context_before if context_before else None,
            context_after=context_after if context_after else None
        )
        
        print("\n📊 LLM RESPONSE:")
        print(json.dumps(llm_result, indent=2))
        
        print("\n" + "=" * 100)
        print(f"FINISHED SEGMENT {segment_id}")
        print("=" * 100)
        
        # Add a pause between segments for readability
        input("\nPress Enter to continue to next segment (or Ctrl+C to stop)...")

if __name__ == "__main__":
    json_path = 'data/augmented_texts.json'
    
    print("\n🎯 This script will process segments one by one and show you:")
    print("   1. What context is extracted")
    print("   2. The EXACT prompt sent to the LLM (including system prompt)")
    print("   3. The LLM's response")
    print("\n" + "=" * 100)
    
    try:
        test_with_detailed_logging(json_path, num_segments=5)
        print("\n✅ Test completed!")
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
