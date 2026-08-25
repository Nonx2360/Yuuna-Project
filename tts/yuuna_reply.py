import json
import re
from typing import Optional

from pydantic import BaseModel, ValidationError


class YuunaReply(BaseModel):
    text: str
    emotion: str = "neutral"
    intensity: float = 0.5


EMOTION_STYLE_MAP = {
    "neutral": "speak in a calm, neutral tone",
    "excited": "speak with excitement and energy",
    "cheerful": "speak cheerfully and warmly",
    "sad": "speak softly with a hint of sadness",
    "shy": "speak hesitantly and a bit shyly",
    "annoyed": "speak with mild annoyance, slightly clipped",
    "sleepy": "speak slowly, softly, as if sleepy",
}

# Map JSON emotions to existing VTS hotkey tags
EMOTION_TO_VTS_TAG = {
    "neutral": "CALM",
    "excited": "EXCITED",
    "cheerful": "HAPPY",
    "sad": "SAD",
    "shy": "SHY",
    "annoyed": "PLAYFUL",
    "sleepy": "CALM",
}


def _extract_json_object(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def parse_gemma_output(raw: str) -> YuunaReply:
    try:
        data = _extract_json_object(raw)
        if data is not None:
            return YuunaReply(**data)
    except ValidationError:
        pass

    clean = re.sub(r"\[[A-Z]+\]\s*", "", raw).strip()
    return YuunaReply(text=clean or raw.strip())


def build_style_prompt(reply: YuunaReply) -> str:
    base = EMOTION_STYLE_MAP.get(reply.emotion.lower(), EMOTION_STYLE_MAP["neutral"])
    if reply.intensity >= 0.8:
        base += ", quite strongly"
    return base


PROSODY_HINTS = {
    "excited": "!",
    "cheerful": "!",
    "annoyed": "!",
    "sad": "...",
    "shy": "...",
    "sleepy": "...",
    "neutral": "",
}


def apply_prosody_hint(reply: YuunaReply) -> str:
    """Return reply.text with a punctuation-only prosody hint.

    The Base TTS model speaks any prepended instruction aloud (it has no
    instruct channel like the VoiceDesign/CustomVoice variants), so emotion
    must be conveyed through punctuation instead of style text.
    """
    tail = PROSODY_HINTS.get(reply.emotion.lower(), "")
    text = reply.text.rstrip()
    if tail and not text.endswith(("!", "?", ".", "...")):
        return f"{text}{tail}"
    return reply.text


def format_display_text(reply: YuunaReply) -> str:
    vts_tag = EMOTION_TO_VTS_TAG.get(reply.emotion.lower(), "CALM")
    return f"[{vts_tag}] {reply.text}"
