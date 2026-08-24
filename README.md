# Yuuna Project 🌸

**Yuuna** is a fully-local, emotion-aware **AI VTuber companion**. It chains together four subsystems into one real-time pipeline:

```
 🎤 Microphone          🧠 Brain                    🔊 Voice              🎭 Avatar
 ─────────────    →    ──────────────────    →    ────────────────    →    ───────────────
 Whisper STT           Gemma 4 E4B (4-bit)        Qwen3-TTS 1.7B        VTube Studio API
 (user_mic/)           structured JSON out         voice cloning         hotkeys / params
                       + emotion + intensity       (tts/qwen_handler)    (vts_connector.py)
```

Everything runs on your own machine — no cloud APIs, no per-token costs, no data leaving your GPU.

---

## ✨ Features

| | Feature | Details |
|---|---|---|
| 🧠 | **Local LLM brain** | Google **Gemma 4 E4B-it**, loaded via `AutoModelForImageTextToText` with optional **NF4 4-bit quantization** (BitsAndBytes) so it fits an 8 GB GPU |
| 💬 | **Structured emotional output** | Yuna replies in strict JSON: `{"text": "...", "emotion": "...", "intensity": 0.0–1.0}` — validated with **Pydantic**, with automatic fallback parsing if the model emits plain text or legacy `[TAG]` markers |
| 🔊 | **Voice-cloned TTS** | **Qwen3-TTS-12Hz-1.7B-Base** clones Yuuna's voice from a single reference clip (`yuuna_ref.wav` + its transcript), then speaks each line with a natural-language *style prompt* derived from the emotion |
| 🎤 | **Speech-to-text** | **OpenAI Whisper** (`user_mic/stt_handler.py`) with a browser-driven recorder (`user_mic/recorder.py`) |
| 🎭 | **Live avatar control** | Full **VTube Studio** WebSocket plugin integration — auth/token persistence, hotkey listing & triggering, parameter injection, and automatic **emotion → hotkey mapping** |
| 🌐 | **Modern web UI** | Flask + Server-Sent Events for **real-time streaming** responses, character switcher, mic controls, VTS control panel |
| 👥 | **Multi-character system** | Create/edit/delete characters in `characters.json`; the default character reads her persona live from `system_prompt.md`; built-in AI-assisted system-prompt generator |
| ⚡ | **Performance-minded loading** | `low_cpu_mem_usage`, bfloat16 on CUDA, graceful CPU fallback, and verbose error reporting on model load failures |

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Web**: Flask, Flask-CORS, HTML5 / CSS3 / vanilla ES6+
- **LLM**: Transformers ≥ 4.57 (`AutoModelForImageTextToText`), Accelerate, BitsAndBytes
- **TTS**: `qwen-tts` (Qwen3-TTS), torchaudio, soundfile
- **STT**: openai-whisper, PyAudio
- **Avatar**: websocket-client against the VTube Studio plugin API (port `8001`)
- **Validation**: Pydantic

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (project targets ~8 GB VRAM, e.g. RTX 4060); CPU mode works but is slow
- [VTube Studio](https://denchisoft.com/) *(optional — avatar control)*
- A clean **3–10 s mono reference voice clip** of the character you want to clone (one is bundled at `TTS-Start-here/yuuna_ref.wav`)

### 2. Model Files

Place the models under `Models_Files/` (paths are configurable in code):

```
Models_Files/
├── google-gemma-4-E4B-it/        # LLM brain
└── Qwen3-TTS-12Hz-1.7B-Base/     # TTS engine
```

### 3. Installation

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell
source venv/bin/activate         # Linux/macOS

# Core dependencies (Flask + LLM + TTS + STT)
pip install -r requirements.txt

# Optional: isolated environments
pip install -r requirements-tts.txt   # TTS-only environment
pip install -r requirements-web.txt   # full web-stack pinning
```

### 4. Running

```powershell
python app.py          # Web interface  → http://localhost:5000
python chat.py         # Terminal chat (no web UI)
.\run.ps1              # Helper launcher script
```

Quick sanity check for the reply-parsing layer (no models required):

```powershell
python TTS-Start-here\test_parse.py
```

---

## 🧩 How the Pipeline Works

1. **You speak** — the browser captures mic audio; `/api/stt` transcribes it with Whisper.
2. **Yuuna thinks** — the text is sent to Gemma 4 E4B with Yuuna's persona (`system_prompt.md`). The system prompt instructs the model to answer **only** with valid JSON:
   ```json
   {"text": "<the spoken reply>", "emotion": "<neutral|excited|cheerful|sad|shy|annoyed|sleepy>", "intensity": <0.0–1.0>}
   ```
3. **Response streams** — tokens stream to the browser over SSE while generation runs on a background thread (`TextIteratorStreamer` + custom stop-criteria).
4. **Parsing layer** (`tts/yuuna_reply.py`) — validates the JSON with Pydantic, strips markdown fences, extracts embedded JSON objects as a fallback, and finally falls back to legacy `[EMOTION]` tag stripping.
5. **Voice** — `tts/qwen_handler.py` converts the emotion into a natural-language style instruction (e.g. *"speak cheerfully and warmly"*), prepends it to the text, and generates audio by **voice cloning** the reference clip. Returns WAV bytes to `/api/tts`.
6. **Avatar** — the emotion is mapped onto VTS tags and forwarded to VTube Studio:

   | Model emotion | VTS hotkey tag |
   |---|---|
   | neutral / sleepy | `CALM` |
   | excited | `EXCITED` |
   | cheerful | `HAPPY` |
   | sad | `SAD` |
   | shy | `SHY` |
   | annoyed | `PLAYFUL` |

---

## 📁 Project Structure

```
Yuuna-Project/
├── app.py                     # Main Flask app: routes, model loading, streaming chat
├── chat.py                    # Standalone CLI chat client
├── vts_connector.py           # VTube Studio WebSocket client (auth, hotkeys, params)
├── characters.json            # Character definitions
├── vts_mappings.json          # Saved emotion → hotkey mappings
├── .vts_token                 # Persisted VTube Studio auth token (auto-generated)
├── system_prompt.md           # Live persona/system prompt for the default character
├── run.ps1                    # Launcher helper (Windows)
│
├── tts/
│   ├── yuuna_reply.py         # Pydantic schema + JSON/tag parsing + emotion maps
│   └── qwen_handler.py        # Qwen3-TTS loader & voice-clone WAV synthesis
│
├── user_mic/
│   ├── stt_handler.py         # Whisper transcription wrapper
│   └── recorder.py            # Microphone capture helpers
│
├── TTS-Start-here/
│   ├── yuuna_ref.wav          # Reference voice clip for cloning
│   ├── transcript-form-ref_audio.txt  # Transcript of the reference clip
│   ├── plan.md                # Design notes for the TTS integration
│   └── test_parse.py          # Offline test for the reply parser
│
├── Models_Files/              # Local model weights (not committed)
├── templates/                 # index.html (chat UI), vts_test.html (VTS console)
├── static/                    # css/, js/, img/
├── dataset.jsonl              # Fine-tuning dataset
└── new-ds/                    # Dataset tooling & improved datasets
```

---

## ⚙️ Configuration

Key knobs at the top of `app.py`:

| Constant | Default | Purpose |
|---|---|---|
| `BASE_MODEL_PATH` | `Models_Files/google-gemma-4-E4B-it` | LLM weights location |
| `VTS_HOST` / `VTS_PORT` | `127.0.0.1` / `8001` | VTube Studio plugin API |
| `SYSTEM_PROMPT_FILE` | `system_prompt.md` | Persona source (re-read every request for the default char) |

In `tts/qwen_handler.py`:

| Setting | Default | Purpose |
|---|---|---|
| `QWEN_TTS_MODEL` (env var) | `Models_Files/Qwen3-TTS-12Hz-1.7B-Base` | Override the TTS model path |
| `REF_AUDIO` / `REF_TEXT_FILE` | `TTS-Start-here/yuuna_ref.wav` + transcript | Voice-clone identity |

---

## 🎯 API Endpoints

### Chat & Speech
| Method | Route | Description |
|---|---|---|
| `POST` | `/api/chat` | Streamed chat completion (SSE) |
| `POST` | `/api/stt` | Transcribe uploaded audio via Whisper |
| `POST` | `/api/tts` | Synthesize speech for a reply → returns WAV |
| `GET` | `/api/tts/status` | TTS engine readiness |

### Characters
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/characters` | List characters |
| `POST` | `/api/characters` | Create character |
| `DELETE` | `/api/characters/<id>` | Delete character |
| `POST` | `/api/generate_prompt` | AI-generate a system prompt from instructions |

### VTube Studio
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/vts/config` | Connection status |
| `POST` | `/api/vts/connect` | Authenticate (token persisted to `.vts_token`) |
| `POST` | `/api/vts/clear_token` | Forget stored token |
| `GET` | `/api/vts/hotkeys` | List available hotkeys |
| `POST` | `/api/vts/trigger` | Fire a hotkey by ID |
| `POST` | `/api/vts/parameter` | Inject a parameter value |
| `GET`/`POST` | `/api/vts/mapping` | Read/save emotion→hotkey mappings |

### Pages
| Route | Description |
|---|---|
| `/` | Main chat interface |
| `/vts_test` | VTube Studio control/test console |

---

## 🧪 Troubleshooting

- **Model fails to load / OOM** — the loader prints a full traceback now instead of dying silently. Try disabling quantization or switching to CPU (`DEVICE` auto-selects; force it by editing `DEVICE` in `app.py`).
- **Transformers version error** (`AutoModelForMultimodalLM` not found) — upgrade: `transformers>=4.57`. The project uses the newer `AutoModelForImageTextToText` API and the `dtype=` argument.
- **No voice output** — ensure `TTS-Start-here/yuuna_ref.wav` and its transcript exist; check `/api/tts/status`.
- **VTS won't connect** — launch VTube Studio, enable the plugin API, then call `/api/vts/connect` once and approve the popup.

---

## 📄 License

MIT — see [LICENSE](LICENSE). Feel free to explore and modify!

## 🤝 Contributing

Issues and enhancement requests are welcome.
