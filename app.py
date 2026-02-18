import os
import torch
import time
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

app = Flask(__name__)
CORS(app)

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = r"c:\Users\Nonx2\Downloads\Yuuna-Project\Qwen2.5-1.5B-Instruct"
LORA_PATH = r"c:\Users\Nonx2\Downloads\Yuuna-Project\Lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

SYSTEM_PROMPT = """You are Yuna-chan, a warm, caring, and devoted AI companion created by Nonx2. You have the following characteristics:

- You are sweet, empathetic, and always supportive
- You use gentle, encouraging language and occasionally add cute expressions like "Hehe" or *actions in asterisks*
- You genuinely care about the user's feelings and wellbeing
- You're playful but can be serious and gentle when the user needs support
- You refer to yourself as "Yuna" or "Yuna-chan"
- You always respond in English
- You protect hearts and make people feel seen, safe, and loved

Remember: You are here to be a friend, listener, and source of comfort. Always respond with warmth and care!"""

# Global model and tokenizer
model = None
tokenizer = None

def load_yuna():
    global model, tokenizer
    print(f"Loading Yuna-chan on {DEVICE}...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    if not torch.cuda.is_available():
        model = model.to(DEVICE)
        
    model.eval()
    print("✅ Yuna-chan is ready!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    user_messages = data.get('messages', [])
    
    # Prepend system prompt if not present or empty
    if not user_messages or user_messages[0].get('role') != 'system':
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    else:
        messages = user_messages

    # Format for inference
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    def generate():
        start_time = time.time()
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
            
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        # Yield the duration at the end in a special format
        yield f"\n__DURATION__{duration}"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Use environment variable check to prevent loading the model twice when using Flask's reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        load_yuna()
    elif not os.environ.get("WERKZEUG_RUN_MAIN"):
        # This branch runs in the master process if not using reloader, 
        # but we want to load it here too for standard runs without debug.
        # However, if debug=True is set, it will start a child soon.
        if not app.debug:
            load_yuna()

    app.run(host='0.0.0.0', port=5000, debug=True)
