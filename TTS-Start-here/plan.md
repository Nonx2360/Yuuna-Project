# Yuuna-chan × Qwen3-TTS Integration Plan

**Goal:** Wire Qwen3-TTS into the existing Yuuna-chan FastAPI + Gemma stack, with emotion-aware style tags and low-latency audio streaming.

---

## 0. Prerequisites

- Existing Yuuna-chan FastAPI backend running Gemma 
- RTX 4060 (8GB VRAM) — sufficient for Qwen3-TTS-12Hz-1.7B
- A clean 3–10s reference voice clip for Yuuna (`yuuna_ref.wav`, 16kHz or 24kHz mono)
- Python venv with Qwen3-TTS installed (see previous setup step)

---

## 1. Style-tag prompting on the Gemma side

Have Gemma emit structured JSON instead of raw text, so emotion metadata flows to TTS for free.

### 1.1 System prompt addition for Gemma

Append to Yuuna's existing system prompt:

```
When responding as Yuuna, always output valid JSON in this exact schema:
{
  "text": "<the spoken reply in Japanese/Thai/English>",
  "emotion": "<one of: neutral, excited, cheerful, sad, shy, annoyed, sleepy>",
  "intensity": <float 0.0-1.0>
}
Do not include any text outside the JSON object.
```

### 1.2 Parsing layer (FastAPI, before hitting TTS)

```python
import json
from pydantic import BaseModel, ValidationError

class YuunaReply(BaseModel):
    text: str
    emotion: str = "neutral"
    intensity: float = 0.5

def parse_gemma_output(raw: str) -> YuunaReply:
    try:
        data = json.loads(raw)
        return YuunaReply(**data)
    except (json.JSONDecodeError, ValidationError):
        # Fallback: treat raw output as plain text, neutral emotion
        return YuunaReply(text=raw.strip())
```

### 1.3 Emotion → style string mapping

Qwen3-TTS takes natural language style instructions, not fixed enum tags — translate:

```python
EMOTION_STYLE_MAP = {
    "neutral":  "speak in a calm, neutral tone",
    "excited":  "speak with excitement and energy",
    "cheerful": "speak cheerfully and warmly",
    "sad":      "speak softly with a hint of sadness",
    "shy":      "speak hesitantly and a bit shyly",
    "annoyed":  "speak with mild annoyance, slightly clipped",
    "sleepy":   "speak slowly, softly, as if sleepy",
}

def build_style_prompt(reply: YuunaReply) -> str:
    base = EMOTION_STYLE_MAP.get(reply.emotion, EMOTION_STYLE_MAP["neutral"])
    if reply.intensity >= 0.8:
        base += ", quite strongly"
    return base
```

**Agent task:** wire `parse_gemma_output` + `build_style_prompt` into the existing chat handler, right after the call returns.

---

## 2. Audio streaming (avoid waiting for full file write)

Two options depending on how the frontend consumes audio. Recommend **Option A** for a web/browser client (Yuuna-chan's likely Alpine.js/Chrome-extension surface).

### Option A: Chunked HTTP streaming response

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io
import soundfile as sf

app = FastAPI()
model = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B")  # load once at startup

@app.post("/tts/stream")
async def generate_speech_stream(req: TTSRequest):
    reply = parse_gemma_output(req.raw_gemma_output)
    style = build_style_prompt(reply)

    # If the model supports streaming generation, use its stream API directly.
    # Otherwise: generate full audio in-memory, then stream it in chunks.
    audio_array = model.clone_and_generate(
        text=reply.text,
        reference_audio="yuuna_ref.wav",
        style=style,
    )

    buf = io.BytesIO()
    sf.write(buf, audio_array, samplerate=24000, format="WAV")
    buf.seek(0)

    def chunk_generator(chunk_size: int = 4096):
        while chunk := buf.read(chunk_size):
            yield chunk

    return StreamingResponse(chunk_generator(), media_type="audio/wav")
```

Check the Qwen3-TTS repo for a native `stream=True` generation flag — the 12Hz tokenizer architecture is built for streaming synthesis, so a true incremental generator (yield audio chunks as they're produced, not post-hoc chunking) may already exist. That's strictly better than the fallback above — **agent should check repo docs/examples for a streaming generation API before falling back to post-hoc chunking.**

### Option B: WebSocket for real-time chat (lower latency, more setup)

If Yuuna-chan's frontend already has a WS connection (check the browser extension code), extend it:

```python
from fastapi import WebSocket

@app.websocket("/ws/tts")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        raw = await websocket.receive_text()
        reply = parse_gemma_output(raw)
        style = build_style_prompt(reply)

        # Stream audio chunks as they're generated
        async for audio_chunk in model.stream_generate(
            text=reply.text,
            reference_audio="yuuna_ref.wav",
            style=style,
        ):
            await websocket.send_bytes(audio_chunk)
```

`model.stream_generate` is illustrative — **agent must confirm actual streaming method name/signature from the Qwen3-TTS repo**, this may differ.

---

## 3. Integration checklist for the AI agent

- [ ] Confirm exact Qwen3-TTS Python API (`clone_and_generate` name/signature may differ from README examples — check `examples/` in the repo)
- [ ] Confirm whether native streaming generation exists (`stream=True` or similar) before implementing Option A's chunking fallback
- [ ] Add `parse_gemma_output` + `build_style_prompt` into existing chat route
- [ ] Update Gemma system prompt to emit the JSON schema in section 1.1
- [ ] Add fallback handling: if Gemma doesn't return valid JSON, degrade gracefully to neutral-tone plain text (already handled in `parse_gemma_output`)
- [ ] Load Qwen3-TTS model once at FastAPI startup (`@app.on_event("startup")` or lifespan handler), not per-request
- [ ] Test reference voice clip quality — iterate on `yuuna_ref.wav` before wiring the full pipeline
- [ ] Decide Option A vs B based on existing frontend transport (check browser extension / Alpine.js code for WS vs REST)
- [ ] Add basic error handling: TTS generation failure should not crash the chat turn — fall back to text-only reply

---

## 4. Open questions to resolve before/during implementation

1. Does the Yuuna-chan frontend currently play audio via `<audio>` tag src, MediaSource API, or Web Audio API? This determines whether Option A or B is the right transport.
2. What sample rate does the Qwen3-TTS output default to — confirm against what the frontend audio player expects (resample if needed).
3. Is there a target latency budget (e.g. "audio should start within Xms of Gemma's reply")? This determines whether the streaming complexity of Option B is worth it vs simpler Option A.