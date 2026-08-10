import io
import os
import threading
from typing import Tuple

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from tts.yuuna_reply import YuunaReply, build_style_prompt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO = os.path.join(BASE_DIR, "TTS-Start-here", "yuuna_ref.wav")
REF_TEXT_FILE = os.path.join(BASE_DIR, "TTS-Start-here", "transcript-form-ref_audio.txt")
MODEL_ID = os.environ.get(
    "QWEN_TTS_MODEL",
    os.path.join(BASE_DIR, "Models_Files", "Qwen3-TTS-12Hz-1.7B-Base"),
)

_handler = None
_handler_lock = threading.Lock()


def _read_ref_text() -> str:
    if os.path.exists(REF_TEXT_FILE):
        with open(REF_TEXT_FILE, "r", encoding="utf-8") as f:
            return " ".join(line.strip() for line in f if line.strip())
    return ""


class QwenTTSHandler:
    def __init__(self):
        self.model = None
        self.voice_clone_prompt = None
        self.ready = False
        self.error = None
        self.sample_rate = 24000

    def load(self):
        if not os.path.exists(REF_AUDIO):
            raise FileNotFoundError(f"Reference audio not found: {REF_AUDIO}")

        ref_text = _read_ref_text()
        if not ref_text:
            raise ValueError(f"Reference transcript is empty: {REF_TEXT_FILE}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        kwargs = {
            "device_map": device,
            "dtype": dtype,
        }
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
            self.model = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)
        except Exception:
            kwargs.pop("attn_implementation", None)
            self.model = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)

        self.voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        self.ready = True

    def _styled_text(self, reply: YuunaReply) -> str:
        style = build_style_prompt(reply)
        return f"({style}) {reply.text}"

    def generate_wav_bytes(self, reply: YuunaReply) -> Tuple[bytes, int]:
        if not self.ready or self.model is None:
            raise RuntimeError(self.error or "Qwen3-TTS is not loaded")

        wavs, sr = self.model.generate_voice_clone(
            text=self._styled_text(reply),
            language="English",
            voice_clone_prompt=self.voice_clone_prompt,
            non_streaming_mode=True,
        )
        self.sample_rate = sr

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)
        return buf.read(), sr

    def iter_wav_chunks(self, reply: YuunaReply, chunk_size: int = 4096):
        wav_bytes, _ = self.generate_wav_bytes(reply)
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i : i + chunk_size]


def get_tts_handler() -> QwenTTSHandler:
    global _handler
    with _handler_lock:
        if _handler is None:
            _handler = QwenTTSHandler()
        return _handler
