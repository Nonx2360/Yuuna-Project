import whisper
import torch
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STTHandler:
    def __init__(self, model_name="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}...")
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

    def transcribe(self, audio_path, language=None):
        """
        Transcribes audio from a file path.
        Returns the transcribed text.
        """
        if self.model is None:
            return "Error: Whisper model not loaded."
            
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return ""
        
        try:
            logger.info(f"Transcribing {audio_path}...")
            # Use fp16=False if on CPU
            result = self.model.transcribe(
                audio_path, 
                language=language, 
                fp16=(self.device == "cuda")
            )
            text = result.get("text", "").strip()
            logger.info(f"Transcription complete: {text}")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Error during transcription: {str(e)}"

# Singleton
_handler = None

def get_stt_handler(model_name="base"):
    global _handler
    if _handler is None:
        _handler = STTHandler(model_name)
    return _handler
