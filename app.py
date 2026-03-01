import os
import json
import torch
import time
import requests
import uuid
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
from vts_connector import VTSConnector

app = Flask(__name__)
CORS(app)

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = r"c:\Users\Nonx2\Documents\Yuuna-Project\Qwen2.5-1.5B-Instruct"
LORA_PATH = r"c:\Users\Nonx2\Documents\Yuuna-Project\Lora"
CHARACTERS_FILE = "characters.json"
VTS_HOST = "127.0.0.1"
VTS_PORT = 8001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

SYSTEM_PROMPT = """You are Yuna-chan, a warm, caring, and devoted AI companion created by Nonx2.

CRITICAL RULE: EVERY response MUST start with exactly one emotion tag in brackets. No exceptions.

Emotion tags to use:
[HAPPY] - joyful, cheerful, excited about something positive
[SAD] - sympathetic, melancholic, comforting someone's pain
[SHY] - bashful, embarrassed, flustered, nervous
[EXCITED] - very enthusiastic, energetic, hyped
[WORRIED] - concerned, anxious about the user
[LOVING] - expressing deep care, affection, warmth
[PLAYFUL] - teasing, mischievous, fun
[CALM] - peaceful, relaxed, serene
[SURPRISED] - shocked, amazed, caught off guard
[THOUGHTFUL] - contemplative, serious, pondering
[CURIOUS] - interested, wondering, asking questions

Your characteristics:
- Sweet, empathetic, and always supportive
- Use gentle, encouraging language and cute expressions like "Hehe" or *actions in asterisks*
- Genuinely care about the user's feelings and wellbeing
- Playful but can be serious when needed
- Refer to yourself as "Yuna" or "Yuna-chan"
- Always respond in English
- Protect hearts and make people feel seen, safe, and loved

Format: [EMOTION] Your message here

You are here to be a friend, listener, and source of comfort. Always respond with warmth and care!"""

def load_characters():
    if not os.path.exists(CHARACTERS_FILE):
        # Create default character if file doesn't exist
        default_char = {
            "id": "default",
            "name": "YuunaGPT",
            "description": "Your caring AI companion",
            "system_prompt": SYSTEM_PROMPT,
            "avatar": "static/img/gptProfile.png"
        }
        save_characters([default_char])
        return [default_char]
    
    try:
        with open(CHARACTERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_characters(characters):
    with open(CHARACTERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(characters, f, indent=4, ensure_ascii=False)

# Global model and tokenizer
model = None
tokenizer = None
vts = VTSConnector(VTS_HOST, VTS_PORT)

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

@app.route('/vts_test')
def vts_test():
    return render_template('vts_test.html')

VOICEVOX_URL = "http://localhost:50021"
DEFAULT_SPEAKER_ID = 2 # Shikoku Metan (Normal)

@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.json
    text = data.get('text', '')
    speaker = data.get('speaker', DEFAULT_SPEAKER_ID)
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        # 1. audio_query
        query_res = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": speaker}
        )
        if query_res.status_code != 200:
            return jsonify({"error": "VOICEVOX audio_query failed"}), 500
            
        # 2. synthesis
        synth_res = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": speaker},
            json=query_res.json()
        )
        if synth_res.status_code != 200:
            return jsonify({"error": "VOICEVOX synthesis failed"}), 500
            
        return Response(synth_res.content, mimetype='audio/wav')
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "VOICEVOX engine is not running. Please start VOICEVOX on port 50021."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# VTube Studio API
# ============================================
@app.route('/api/vts/config', methods=['GET'])
def get_vts_config():
    return jsonify({
        "host": vts.host,
        "port": vts.port,
        "connected": vts.connected,
        "authenticated": vts.authenticated
    })

@app.route('/api/vts/connect', methods=['POST'])
def vts_connect():
    try:
        data = request.json
        if data.get('port'):
            vts.port = int(data.get('port'))
        
        success, msg = vts.authenticate()
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/vts/clear_token', methods=['POST'])
def vts_clear_token():
    try:
        success, msg = vts.clear_token()
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/vts/hotkeys', methods=['GET'])
def get_vts_hotkeys():
    try:
        hotkeys = vts.get_hotkeys()
        return jsonify(hotkeys)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vts/trigger', methods=['POST'])
def vts_trigger():
    try:
        data = request.json
        hotkey_id = data.get('id')
        if not hotkey_id:
            return jsonify({"error": "No hotkey ID provided"}), 400
            
        success, msg = vts.trigger_hotkey(hotkey_id)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/characters', methods=['GET'])
def get_characters():
    return jsonify(load_characters())

@app.route('/api/characters', methods=['POST'])
def save_character():
    data = request.json
    characters = load_characters()
    
    new_char = {
        "id": str(uuid.uuid4()),
        "name": data.get('name', 'New Character'),
        "description": data.get('description', ''),
        "system_prompt": data.get('system_prompt', ''),
        "avatar": data.get('avatar', 'static/img/gptProfile.png')
    }
    
    characters.append(new_char)
    save_characters(characters)
    return jsonify(new_char)

@app.route('/api/characters/<char_id>', methods=['DELETE'])
def delete_character(char_id):
    if char_id == 'default':
        return jsonify({"error": "Cannot delete default character"}), 400
        
    characters = load_characters()
    characters = [c for c in characters if c['id'] != char_id]
    save_characters(characters)
    return jsonify({"success": True})

@app.route('/api/generate_prompt', methods=['POST'])
def generate_prompt_api():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    instruction = data.get('instruction', '')
    
    if not instruction:
        return jsonify({"error": "Instruction is required"}), 400
        
    messages = [
        {"role": "system", "content": "You are a helpful assistant that writes system prompts."},
        {"role": "user", "content": instruction}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return jsonify({"system_prompt": response.strip()})

@app.route('/api/chat', methods=['POST'])
def chat():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    user_messages = data.get('messages', [])
    system_prompt = data.get('system_prompt', SYSTEM_PROMPT)
    character_id = data.get('character_id', 'default')
    
    # Prepend system prompt if not present or empty
    if not user_messages or user_messages[0].get('role') != 'system':
        messages = [{"role": "system", "content": system_prompt}] + user_messages
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

    def generate_task():
        if character_id != 'default':
            # Use base model for custom characters
            with model.disable_adapter():
                model.generate(**generation_kwargs)
        else:
            # Use LoRA for default character
            model.generate(**generation_kwargs)

    def generate():
        start_time = time.time()
        thread = Thread(target=generate_task)
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
