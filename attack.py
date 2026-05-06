import json
import time
from deep_translator import GoogleTranslator

# Load your JSON data
input_file = 'results/T5_C4_5k_SIR_1.0_20260204.json'
output_file = 'attacked_data_sir_1.0.json'

def translation_attack(text, pivot_language='fr'):
    """
    Translates text to a pivot language and back to English
    to disrupt watermark signals.
    """
    if not text:
        return ""
    
    try:
        # Step 1: English -> Pivot (e.g., French)
        translated = GoogleTranslator(source='en', target=pivot_language).translate(text)
        
        # Step 2: Pivot -> English
        back_translated = GoogleTranslator(source=pivot_language, target='en').translate(translated)
        
        return back_translated
    except Exception as e:
        print(f"Error processing text: {e}")
        return text

def process_data(data):
    # Iterate through the JSON structure
    for entry in data:
        if 'samples' in entry:
            for sample in entry['samples']:
                original_wm = sample.get('watermarked', '')
                
                print(f"Attacking sample: {sample.get('prompt', '')[:30]}...")
                
                # Perform attack
                attacked_text = translation_attack(original_wm)
                
                # Save result
                sample['attacked_text'] = attacked_text
                
                # Optional: Sleep to avoid rate limiting if processing many items
                time.sleep(0.5) 
    return data

# Execution
if __name__ == "__main__":
    # Assuming 'json_data' is your list loaded from file
    with open(input_file, 'r') as f:
        json_data = json.load(f)

    attacked_data = process_data(json_data)

    with open(output_file, 'w') as f:
        json.dump(attacked_data, f, indent=4)
        
    print(f"Attack complete. Data saved to {output_file}")
