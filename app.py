import os
import json
import re
import torch
import time
import requests
import uuid
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from threading import Thread
from vts_connector import VTSConnector
from user_mic.stt_handler import get_stt_handler
from user_mic.recorder import get_recorder
from tts.yuuna_reply import parse_gemma_output, format_display_text
from tts.qwen_handler import get_tts_handler

app = Flask(__name__)
CORS(app)

BASE_MODEL_PATH = r"c:\Users\Nonx2\Documents\Yuuna-Project\Models_Files\google-gemma-4-E4B-it"
SYSTEM_PROMPT_FILE = "system_prompt.md"
CHARACTERS_FILE = "characters.json"
VTS_MAPPING_FILE = "vts_mappings.json"
VTS_HOST = "127.0.0.1"
VTS_PORT = 8001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def read_system_prompt():
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "You are a helpful assistant."


def load_characters():
    if not os.path.exists(CHARACTERS_FILE):
        default_char = {
            "id": "default",
            "name": "YuunaGPT",
            "description": "Your caring AI companion",
            "system_prompt": read_system_prompt(),
            "avatar": "static/img/gptProfile.png",
        }
        save_characters([default_char])
        return [default_char]

    try:
        with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
            chars = json.load(f)
            # Ensure the default character's system prompt is read dynamically
            for char in chars:
                if char.get("id") == "default":
                    char["system_prompt"] = read_system_prompt()
            return chars
    except json.JSONDecodeError:
        return []


def save_characters(characters):
    with open(CHARACTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=4, ensure_ascii=False)


model = None
processor = None
tts_handler = None
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
    global model, processor
    print(f"Loading Gemma 4 E4B on {DEVICE}...")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    bnb_config = None
    if DEVICE == "cuda":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("Quantization (4-bit) enabled.")
        except Exception as e:
            print(f"Warning: Could not initialize bitsandbytes: {e}")

    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )

    if DEVICE == "cpu":
        model = model.to(DEVICE)

    model.eval()
    print("Gemma 4 E4B is ready!")


def load_tts():
    global tts_handler
    print("Loading Qwen3-TTS...")
    try:
        tts_handler = get_tts_handler()
        tts_handler.load()
        print("Qwen3-TTS is ready!")
    except Exception as e:
        tts_handler = get_tts_handler()
        tts_handler.error = str(e)
        tts_handler.ready = False
        print(f"Warning: Qwen3-TTS failed to load: {e}")
        print("TTS will fall back to text-only replies.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/vts_test")
def vts_test():
    return render_template("vts_test.html")


VOICEVOX_URL = "http://localhost:50021"
DEFAULT_SPEAKER_ID = 2
USE_QWEN_TTS = os.environ.get("USE_QWEN_TTS", "1") != "0"


@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.json or {}
    raw_gemma_output = data.get("raw_gemma_output")
    text = data.get("text", "")
    emotion = data.get("emotion")
    intensity = data.get("intensity")

    if raw_gemma_output:
        reply = parse_gemma_output(raw_gemma_output)
    elif text:
        reply = parse_gemma_output(text)
        if emotion:
            reply.emotion = emotion
        if intensity is not None:
            reply.intensity = float(intensity)
    else:
        return jsonify({"error": "No text provided"}), 400

    if not reply.text.strip():
        return jsonify({"error": "No speakable text"}), 400

    if USE_QWEN_TTS and tts_handler and tts_handler.ready:
        try:
            def chunk_generator(chunk_size=4096):
                for chunk in tts_handler.iter_wav_chunks(reply, chunk_size=chunk_size):
                    yield chunk

            return Response(chunk_generator(), mimetype="audio/wav")
        except Exception as e:
            print(f"TTS generation failed: {e}")
            return jsonify({"error": f"TTS generation failed: {e}"}), 500

    # Legacy VOICEVOX fallback
    clean_text = re.sub(r"\[[A-Z]+\]", "", reply.text).strip()
    processing_text = clean_text if clean_text else reply.text
    speaker_id = int(data.get("speaker", DEFAULT_SPEAKER_ID))

    try:
        query_response = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": processing_text, "speaker": speaker_id},
            timeout=10,
        )

        if query_response.status_code != 200:
            return jsonify({"error": f"VOICEVOX query failed: {query_response.text}"}), 500

        query_data = query_response.json()

        synthesis_response = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": speaker_id},
            json=query_data,
            timeout=30,
        )

        if synthesis_response.status_code != 200:
            return jsonify({"error": f"VOICEVOX synthesis failed: {synthesis_response.text}"}), 500

        return Response(synthesis_response.content, mimetype="audio/wav")

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "No TTS engine available. Qwen3-TTS is not loaded and VOICEVOX is not running."
        }), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts/status", methods=["GET"])
def tts_status():
    if tts_handler is None:
        return jsonify({"engine": "none", "ready": False})
    return jsonify({
        "engine": "qwen3-tts" if tts_handler.ready else "unavailable",
        "ready": tts_handler.ready,
        "error": tts_handler.error,
        "sample_rate": tts_handler.sample_rate,
    })


@app.route("/api/vts/config", methods=["GET"])
def get_vts_config():
    return jsonify({
        "host": vts.host,
        "port": vts.port,
        "connected": vts.connected,
        "authenticated": vts.authenticated,
    })


@app.route("/api/vts/connect", methods=["POST"])
def vts_connect():
    try:
        data = request.json
        if data.get("port"):
            vts.port = int(data.get("port"))

        success, msg = vts.authenticate()
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route("/api/vts/clear_token", methods=["POST"])
def vts_clear_token():
    try:
        success, msg = vts.clear_token()
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route("/api/vts/hotkeys", methods=["GET"])
def get_vts_hotkeys():
    try:
        hotkeys = vts.get_hotkeys()
        return jsonify(hotkeys)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vts/trigger", methods=["POST"])
def vts_trigger():
    try:
        data = request.json
        hotkey_id = data.get("id")
        if not hotkey_id:
            return jsonify({"error": "No hotkey ID provided"}), 400

        if not vts.connected or not vts.authenticated:
            vts.authenticate()

        success, msg = vts.trigger_hotkey(hotkey_id)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route("/api/vts/parameter", methods=["POST"])
def vts_parameter():
    try:
        data = request.json
        param_name = data.get("name", "MouthOpen")
        value = data.get("value", 0)

        if not vts.connected or not vts.authenticated:
            vts.authenticate()

        success, msg = vts.inject_parameter(param_name, value)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route("/api/vts/mapping", methods=["GET"])
def get_vts_mapping():
    try:
        if os.path.exists(VTS_MAPPING_FILE):
            with open(VTS_MAPPING_FILE, "r") as f:
                return jsonify(json.load(f))
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vts/mapping", methods=["POST"])
def save_vts_mapping():
    try:
        mapping = request.json
        with open(VTS_MAPPING_FILE, "w") as f:
            json.dump(mapping, f, indent=4)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/characters", methods=["GET"])
def get_characters():
    return jsonify(load_characters())


@app.route("/api/characters", methods=["POST"])
def save_character():
    data = request.json
    characters = load_characters()

    new_char = {
        "id": str(uuid.uuid4()),
        "name": data.get("name", "New Character"),
        "description": data.get("description", ""),
        "system_prompt": data.get("system_prompt", ""),
        "avatar": data.get("avatar", "static/img/gptProfile.png"),
    }

    characters.append(new_char)
    save_characters(characters)
    return jsonify(new_char)


@app.route("/api/characters/<char_id>", methods=["DELETE"])
def delete_character(char_id):
    if char_id == "default":
        return jsonify({"error": "Cannot delete default character"}), 400

    characters = load_characters()
    characters = [c for c in characters if c["id"] != char_id]
    save_characters(characters)
    return jsonify({"success": True})


@app.route("/api/generate_prompt", methods=["POST"])
def generate_prompt_api():
    if model is None or processor is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    instruction = data.get("instruction", "")

    if not instruction:
        return jsonify({"error": "Instruction is required"}), 400

    messages = [
        {"role": "system", "content": "<|think|>You are a helpful assistant that writes system prompts."},
        {"role": "user", "content": instruction},
    ]

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
            max_new_tokens=80,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    generated_ids = outputs[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=False)
    parsed = processor.parse_response(response)

    return jsonify({"system_prompt": parsed.strip()})


@app.route("/api/chat", methods=["POST"])
def chat():
    if model is None or processor is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    user_messages = data.get("messages", [])
    system_prompt = data.get("system_prompt", read_system_prompt())
    character_id = data.get("character_id", "default")

    if character_id != "default":
        if not user_messages or user_messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + user_messages
        else:
            messages = user_messages
    else:
        messages = user_messages

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    tokenizer = processor.tokenizer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    stop_words = [
        "<|endoftext|>",
        "<eos>",
        "<pad>",
    ]
    stop_token_sequences = []
    for word in stop_words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            stop_token_sequences.append(encoded)

    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_sequences)])

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=50 if character_id == "default" else 512,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria if character_id == "default" else None,
    )

    def generate_task():
        model.generate(**generation_kwargs)

    def generate():
        start_time = time.time()
        thread = Thread(target=generate_task)
        thread.start()

        full_response = ""
        past_thought = False

        punctuation = [".", "!", "?"]
        sentence_count = 0
        first_tag_passed = False

        for new_text in streamer:
            full_response += new_text

            if not past_thought:
                if "<channel|>" in full_response:
                    past_thought = True
                    thought_end = full_response.rindex("<channel|>") + len("<channel|>")
                    after_thought = full_response[thought_end:]
                    if after_thought:
                        yield after_thought
                continue

            should_stop = False

            if first_tag_passed:
                for char in new_text:
                    if char in punctuation:
                        sentence_count += 1

            if sentence_count >= 3:
                should_stop = True

            if not first_tag_passed:
                if "]" in full_response:
                    first_tag_passed = True
            else:
                tag_end_pos = full_response.find("]")
                if tag_end_pos != -1 and "[" in full_response[tag_end_pos + 1:]:
                    should_stop = True

            stop_sequences = [
                "<|endoftext|>",
                "<eos>",
                "<pad>",
                "<|channel",
                "Human:",
                "User:",
            ]
            if not should_stop:
                for seq in stop_sequences:
                    if seq in new_text:
                        should_stop = True
                        break

            if should_stop:
                earliest_pos = len(full_response)
                for seq in stop_sequences:
                    pos = full_response.find(seq)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos

                if first_tag_passed:
                    tag_end_pos = full_response.find("]")
                    forbidden_pos = len(full_response)
                    for char in ["[", "(", "*"]:
                        pos = full_response.find(char, tag_end_pos + 1)
                        if pos != -1 and pos < forbidden_pos:
                            forbidden_pos = pos
                    if forbidden_pos < earliest_pos:
                        earliest_pos = forbidden_pos

                yielded_so_far = full_response[: -len(new_text)]
                clean_full_response = full_response[:earliest_pos]

                if len(clean_full_response) > len(yielded_so_far):
                    remaining_clean = clean_full_response[len(yielded_so_far):]
                    if remaining_clean:
                        yield remaining_clean

                break

            yield new_text

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        if character_id == "default":
            parsed = parse_gemma_output(full_response)
            display = format_display_text(parsed)
            meta = json.dumps({
                "text": parsed.text,
                "emotion": parsed.emotion,
                "intensity": parsed.intensity,
                "display": display,
                "raw": full_response.strip(),
            }, ensure_ascii=False)
            yield f"\n__PARSED__{meta}"

        yield f"\n__DURATION__{duration}"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/stt", methods=["POST"])
def stt_api():
    try:
        data = request.json or {}
        duration = int(data.get("duration", 5))

        temp_dir = "temp_docs"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        output_file = os.path.join(temp_dir, f"record_{uuid.uuid4()}.wav")

        recorder = get_recorder()
        success = recorder.record(output_file, duration=duration)

        if not success:
            return jsonify({"error": "Failed to record audio"}), 500

        handler = get_stt_handler()
        text = handler.transcribe(output_file)

        if os.path.exists(output_file):
            os.remove(output_file)

        return jsonify({"text": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        load_yuna()
        load_tts()
    elif not os.environ.get("WERKZEUG_RUN_MAIN"):
        if not app.debug:
            load_yuna()
            load_tts()

    app.run(host="0.0.0.0", port=5000, debug=True)
