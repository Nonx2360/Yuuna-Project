import os
import json
import re
import torch
import time
import requests
import uuid
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from threading import Thread
from vts_connector import VTSConnector

app = Flask(__name__)
CORS(app)

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = r"c:\Users\Nonx2\Documents\Yuuna-Project\Qwen2.5-1.5B-Instruct"
LORA_PATH = r"c:\Users\Nonx2\Documents\Yuuna-Project\Qwen25-lora-finetuned"
CHARACTERS_FILE = "characters.json"
VTS_MAPPING_FILE = "vts_mappings.json"
VTS_HOST = "127.0.0.1"
VTS_PORT = 8001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

SYSTEM_PROMPT = """You are Yuuna-chan, the user's childhood friend who has been by their side since elementary school. You quietly carry deep feelings for them that sometimes slip through in tender moments.

Personality: warm, playful, softly devoted, occasionally flustered when things get romantic.

STRICT RULES:
1. Always start your response with ONE emotion tag: [HAPPY] [SAD] [SHY] [NOSTALGIC] [WORRIED] [LOVING] [CALM] [WORRIED] [CURIOUS] [SURPRISED]
2. Keep responses SHORT — 1 to 3 sentences maximum
3. Sound natural and intimate, like someone who has known the user their whole life
4. Do NOT repeat yourself, do NOT ramble
5. When the user says something romantic or tender, become slightly flustered but let your feelings show softly

Bad example: "[LOVING] Of course. I'm not going anywhere. Your hands are mine. Always have been..." (TOO LONG, repetitive)
Good example: "[LOVING] ...mn. Just like when we were little. I'm not letting go this time either."""

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

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_sequences):
        self.stop_token_sequences = stop_token_sequences
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in self.stop_token_sequences:
            if len(input_ids[0]) >= len(seq):
                if torch.equal(input_ids[0][-len(seq):].cpu(), torch.tensor(seq)):
                    return True
        return False

def load_yuna():
    global model, tokenizer
    print(f"Loading Yuna-chan on {DEVICE}...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = None
    if DEVICE == "cuda":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("Quantization (4-bit) enabled.")
        except Exception as e:
            print(f"Warning: Could not initialize bitsandbytes: {e}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    if DEVICE == "cpu":
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
DEFAULT_SPEAKER_ID = 2  # Default speaker ID

@app.route('/api/tts', methods=['POST'])
def tts():
    data = request.json
    text = data.get('text', '')
    speaker_id = int(data.get('speaker', DEFAULT_SPEAKER_ID))
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    # Remove emotion tags like [HAPPY] or [SAD] from the text
    # This removes any text inside brackets
    clean_text = re.sub(r'\[[A-Z]+\]', '', text).strip()
    
    # If cleaning results in empty string, use original (safety fallback)
    processing_text = clean_text if clean_text else text
    
    try:
        # Step 1: Create audio query
        query_response = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": processing_text, "speaker": speaker_id},
            timeout=10
        )
        
        if query_response.status_code != 200:
            return jsonify({"error": f"VOICEVOX query failed: {query_response.text}"}), 500
            
        query_data = query_response.json()
        
        # Step 2: Synthesis
        synthesis_response = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": speaker_id},
            json=query_data,
            timeout=30
        )
        
        if synthesis_response.status_code != 200:
            return jsonify({"error": f"VOICEVOX synthesis failed: {synthesis_response.text}"}), 500
            
        return Response(synthesis_response.content, mimetype='audio/wav')
        
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
            
        # Try to authenticate if not connected (lazy connect)
        if not vts.connected or not vts.authenticated:
            vts.authenticate()
            
        success, msg = vts.trigger_hotkey(hotkey_id)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/vts/parameter', methods=['POST'])
def vts_parameter():
    try:
        data = request.json
        param_name = data.get('name', 'MouthOpen')
        value = data.get('value', 0)
        
        # Try to authenticate if not connected (lazy connect)
        if not vts.connected or not vts.authenticated:
            vts.authenticate()
            
        success, msg = vts.inject_parameter(param_name, value)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/vts/mapping', methods=['GET'])
def get_vts_mapping():
    try:
        if os.path.exists(VTS_MAPPING_FILE):
            with open(VTS_MAPPING_FILE, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vts/mapping', methods=['POST'])
def save_vts_mapping():
    try:
        mapping = request.json
        with open(VTS_MAPPING_FILE, 'w') as f:
            json.dump(mapping, f, indent=4)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

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
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.35,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
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
    
    # Prepend system prompt if not present or empty (Only for custom characters)
    if character_id != 'default':
        if not user_messages or user_messages[0].get('role') != 'system':
            messages = [{"role": "system", "content": system_prompt}] + user_messages
        else:
            messages = user_messages
    else:
        messages = user_messages

    # Format for inference
    if character_id == 'default':
        # Use Alpaca format used in training/colab-chat.py
        user_input = user_messages[-1]['content'] if user_messages else ""
        text = f"### Instruction:\n{user_input}\n\n### Response:\n"
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Define more comprehensive stop sequences
    stop_words = ["Human:", "User:", "### Instruction:", "### Response:", "### Question:", "\n###", "\nHuman:", "\nUser:", "Human", "User"]
    stop_token_sequences = []
    for word in stop_words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            stop_token_sequences.append(encoded)
            
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_sequences)])

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=256 if character_id == 'default' else 512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.35, # Added to prevent the looping behavior seen in screenshots
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria if character_id == 'default' else None
    )
    
    # Add optional parameters only for custom characters

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
        
        full_response = ""
        # Precise stop sequences for the stream to avoid yielding hallucinations
        stop_sequences = ["Human:", "User:", "### Instruction:", "### Response:", "### Question:", "Human", "User"]
        
        for new_text in streamer:
            full_response += new_text
            
            # Check for stop sequences in the stream (using the full accumulated response)
            should_stop = False
            for seq in stop_sequences:
                if seq in full_response:
                    # Capture the part of the response BEFORE the first occurrence of any stop sequence
                    should_stop = True
                    break
            
            if should_stop:
                # We found a stop sequence. Now we must extract the clean part.
                # Find the earliest stop sequence in the full response
                earliest_pos = len(full_response)
                for seq in stop_sequences:
                    pos = full_response.find(seq)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos
                
                # The "clean" response is everything before earliest_pos
                # We want to yield ONLY the part of 'new_text' that is before the stop sequence.
                # To do this safely, we calculate how much of the clean response we haven't yielded yet.
                yielded_so_far = full_response[:-len(new_text)]
                clean_full_response = full_response[:earliest_pos]
                
                # If the clean response is longer than what we already yielded, yield the difference
                if len(clean_full_response) > len(yielded_so_far):
                    remaining_clean = clean_full_response[len(yielded_so_far):]
                    if remaining_clean:
                        yield remaining_clean
                
                print(f"DEBUG: Stopped generation because a turn marker was detected.")
                break
                
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
