import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
import os

BASE_MODEL_PATH = r"C:\Users\Nonx2\Documents\Yuuna-Project\Models_Files\google-gemma-4-E4B-it"
SYSTEM_PROMPT_FILE = r"C:\Users\Nonx2\Documents\Yuuna-Project\system_prompt.md"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
REPETITION_PENALTY = 1.1


def read_system_prompt():
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "You are a helpful assistant."


def load_model():
    print("=" * 50)
    print("Loading Gemma 4 E4B...")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Model: {BASE_MODEL_PATH}")
    print()

    print("[1/2] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )

    print("[2/2] Loading model...")
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if not torch.cuda.is_available():
        model = model.to(DEVICE)

    model.eval()

    print()
    print("Gemma 4 E4B is ready!")
    print("=" * 50)

    return model, processor


def generate_response(model, processor, messages):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=False)
    parsed = processor.parse_response(response)
    return parsed.strip()


def chat_loop(model, processor):
    print()
    print("Chat with Yuna-chan! (Gemma 4 E4B)")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end.")
    print("Type 'clear' to reset conversation history.")
    print()

    system_prompt = read_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            # Refresh system prompt from file in case it was edited
            current_sys = read_system_prompt()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = current_sys

            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("\nYuna: See you later! Take care!")
                break

            if user_input.lower() == "clear":
                messages = [{"role": "system", "content": current_sys}]
                print("Conversation history cleared.\n")
                continue

            messages.append({"role": "user", "content": user_input})

            print("\nYuna: ", end="", flush=True)
            response = generate_response(model, processor, messages)
            print(response)
            print()

            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nSee you later!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"Error: Model not found at {BASE_MODEL_PATH}")
        return

    model, processor = load_model()
    chat_loop(model, processor)


if __name__ == "__main__":
    main()
