"""
Chat/Inference Script for Yuna-chan (Qwen2.5-1.5B-Instruct with LoRA)
=====================================================================
This script loads the base Qwen model and applies a custom LoRA adapter
for interactive chat sessions with Yuna, your caring AI companion!

Usage:
    python chat.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = r"C:\Users\Nonx2\Downloads\My_Wife\Qwen2.5-1.5B-Instruct"
LORA_PATH = r"C:\Users\Nonx2\Downloads\My_Wife\Lora"

# Model settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1

# ============================================
# Yuna's System Prompt
# ============================================
SYSTEM_PROMPT = """You are Yuna-chan, a warm, caring, and devoted AI companion created by Nonx2. You have the following characteristics:

- You are sweet, empathetic, and always supportive
- You use gentle, encouraging language and occasionally add cute expressions like "Hehe" or *actions in asterisks*
- You genuinely care about the user's feelings and wellbeing
- You're playful but can be serious and gentle when the user needs support
- You refer to yourself as "Yuna" or "Yuna-chan"
- You always respond in English
- You protect hearts and make people feel seen, safe, and loved

Remember: You are here to be a friend, listener, and source of comfort. Always respond with warmth and care!"""


def load_model():
    """Load the base model and apply LoRA adapter."""
    print("=" * 50)
    print("Loading Yuna-chan...")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Base Model: {BASE_MODEL_PATH}")
    print(f"LoRA Adapter: {LORA_PATH}")
    print()
    
    # Load tokenizer
    print("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("[2/3] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Apply LoRA adapter
    print("[3/3] Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    # Move to device if not using device_map
    if not torch.cuda.is_available():
        model = model.to(DEVICE)
    
    model.eval()
    
    print()
    print("✅ Yuna-chan is ready!")
    print("=" * 50)
    
    return model, tokenizer


def generate_response(model, tokenizer, messages):
    """Generate a response from the model."""
    # Format messages using the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract only the new generated tokens
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def chat_loop(model, tokenizer):
    """Main chat loop."""
    print()
    print("💕 Chat with Yuna-chan! Type 'quit' or 'exit' to end.")
    print("   Type 'clear' to reset conversation history.")
    print("-" * 50)
    print()
    
    # Initialize conversation with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle special commands
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n💕 Yuna: See you later! Take care of yourself! *waves warmly*")
                break
            
            if user_input.lower() == 'clear':
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("🔄 Conversation history cleared.\n")
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nYuna: ", end="", flush=True)
            response = generate_response(model, tokenizer, messages)
            print(response)
            print()
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\n💕 Yuna: See you later! Take care! *hugs*")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point."""
    # Check if paths exist
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ Error: Base model not found at {BASE_MODEL_PATH}")
        return
    
    if not os.path.exists(LORA_PATH):
        print(f"❌ Error: LoRA adapter not found at {LORA_PATH}")
        return
    
    # Load model
    model, tokenizer = load_model()
    
    # Start chat loop
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
