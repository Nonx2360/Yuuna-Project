# YuunaGPT 🌸

YuunaGPT is a modern, expressive AI companion built for a seamless and high-quality chat experience. It combines a premium dark-themed interface with advanced Text-to-Speech (TTS) capabilities using VOICEVOX to bring your AI to life.

## ✨ Features

- **🚀 Real-time Streaming**: Enjoy lightning-fast, character-by-character responses for a natural conversation flow.
- **🎨 Premium Chat UI**: A sleek, modern dark mode interface inspired by world-class chat applications.
- **🎤 VOICEVOX Integration**: High-quality "Anime Girl" voice output for both Japanese and English.
- **🎭 Emotion Control**: Switch Yuuna's "Voice Mood" on the fly between *Normal*, *Sweet*, *Tsundere*, *Sexy*, and *Whisper*.
- **📝 Markdown Support**: Full support for rich text, including beautiful code highlighting, lists, and bold text.
- **🛠️ Message Actions**: Effortlessly copy messages, rate responses, or trigger voice replay with a single click.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, PyTorch (Transformer-based inference).
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+), Marked.js.
- **TTS Engine**: [VOICEVOX](https://voicevox.hiroshiba.jp/) (Local Engine).

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- [VOICEVOX Engine](https://voicevox.hiroshiba.jp/) (Download and run for TTS support).

### 2. Installation
Clone the repository and install the dependencies:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/scripts/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Running the App
Start the Flask server:
```bash
python app.py
```
Then open your browser and navigate to `http://localhost:5000`.

## 🎤 Setting up TTS (VOICEVOX)
To enable YuunaGPT's voice:
1. Download and install **VOICEVOX** (GPU mode recommended if you have an NVIDIA GPU).
2. Launch the VOICEVOX application.
3. Keep it running in the background while using YuunaGPT.
4. Use the **Auto Voice** toggle in the sidebar to enable/disable speech.

## 📄 License
This project is licensed under the MIT License. Feel free to explore and modify!