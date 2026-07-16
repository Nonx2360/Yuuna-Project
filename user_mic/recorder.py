import pyaudio
import wave
import os
import time
import logging

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000 # Whisper prefers 16kHz
        self.p = pyaudio.PyAudio()

    def record(self, output_path, duration=5):
        """
        Records audio for a fixed duration.
        """
        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            logger.info(f"Start recording to {output_path}...")
            frames = []
            
            for i in range(0, int(self.rate / self.chunk * duration)):
                data = stream.read(self.chunk)
                frames.append(data)
                
            logger.info("Recording complete.")
            
            stream.stop_stream()
            stream.close()
            
            # Save to WAV
            wf = wave.open(output_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return True
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return False

# Singleton
_recorder = None

def get_recorder():
    global _recorder
    if _recorder is None:
        _recorder = AudioRecorder()
    return _recorder
