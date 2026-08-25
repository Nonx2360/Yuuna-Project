"""Standalone TTS test: Qwen3-TTS voice cloning with refaudio.wav.

Follows TTS-Start-here/plan.md:
  raw Gemma-style JSON -> parse_gemma_output() -> build_style_prompt()
  -> Qwen3-TTS generate_voice_clone() -> saved WAV files.

Run with the venv_tts environment:
    ..\venv_tts\Scripts\python.exe test_tts.py
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.yuuna_reply import parse_gemma_output, build_style_prompt, apply_prosody_hint  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

REF_AUDIO = os.path.join(BASE_DIR, "refaudio.wav")
REF_AUDIO_TRIM = os.path.join(BASE_DIR, "refaudio_trim.wav")  # 8.75s mono slice, no ending laughter
REF_TEXT_FILE = os.path.join(BASE_DIR, "transcript-form-ref_audio.txt")
MODEL_ID = os.environ.get(
    "QWEN_TTS_MODEL",
    os.path.join(PROJECT_ROOT, "Models_Files", "Qwen3-TTS-12Hz-1.7B-Base"),
)
OUT_DIR = os.path.join(BASE_DIR, "outputs")

# Transcript for the first ~8.75s of refaudio.wav (matches the trimmed clip)
REF_TRIM_TEXT = "愛してるよ、ユキヤ。ね、ドキドキした？ 私も、愛してるよ、ユキヤ。"

# (raw Gemma JSON, clone mode) — "icl" continues from ref speech codes,
# "xvec" uses speaker embedding only (no transcript continuation artifacts)
TEST_REPLIES_RAW = [
    ('{"text":"Yay! You came back to talk to me! I was starting to get lonely.","emotion":"cheerful","intensity":0.9}', "icl"),
    ('{"text":"Yay! You came back to talk to me! I was starting to get lonely.","emotion":"cheerful","intensity":0.9}', "xvec"),
    ('{"text":"W-wait... you really think my voice sounds cute?","emotion":"shy","intensity":0.85}', "icl"),
    ('{"text":"Oh... you have to go already? Okay... see you later then.","emotion":"sad","intensity":0.6}', "icl"),
]


def main():
    print(f"[1/4] torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[2/4] Loading Qwen3-TTS from {MODEL_ID} ...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
    print(f"      loaded in {time.time() - t0:.1f}s")

    print(f"[3/4] Building voice clone prompts from {REF_AUDIO_TRIM} (trimmed 8.75s, no ending laughter)")
    prompt_icl = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO_TRIM,
        ref_text=REF_TRIM_TEXT,
        x_vector_only_mode=False,
    )
    prompt_xvec = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO_TRIM,
        x_vector_only_mode=True,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[4/4] Generating {len(TEST_REPLIES_RAW)} samples -> {OUT_DIR}")
    for i, (raw, mode) in enumerate(TEST_REPLIES_RAW, 1):
        reply = parse_gemma_output(raw)
        style = build_style_prompt(reply)
        spoken_text = apply_prosody_hint(reply)
        print(f"\n--- Sample {i}: emotion={reply.emotion} intensity={reply.intensity} mode={mode}")
        print(f"    style (info only, NOT spoken): {style}")
        print(f"    speaking     : {spoken_text}")

        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=spoken_text,
            language="Auto",
            voice_clone_prompt=prompt_icl if mode == "icl" else prompt_xvec,
            non_streaming_mode=True,
        )
        out_path = os.path.join(OUT_DIR, f"tts_test_{i}_{reply.emotion}_{mode}.wav")
        sf.write(out_path, np.asarray(wavs[0]), samplerate=sr)
        dur = len(wavs[0]) / sr
        gen_t = time.time() - t0
        print(f"    saved {out_path}")
        print(f"    audio: {dur:.2f}s @ {sr}Hz | generated in {gen_t:.1f}s (RTF {dur and gen_t / dur:.2f}x)")

    print("\nDONE - all samples generated successfully.")


if __name__ == "__main__":
    main()
