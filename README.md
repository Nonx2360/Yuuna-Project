# Yuuna Project

Yuuna Project is a comprehensive AI companion system featuring Yuna-chan, a caring and expressive AI personality. This project combines a locally hosted language model with advanced integrations for voice synthesis, character management, and virtual avatar control.

## ✨ Features

- **🤖 Local AI Model**: Qwen2.5-1.5B-Instruct with custom LoRA fine-tuning for Yuna-chan's personality
- **💬 Emotion-Based Responses**: Yuna-chan responds with emotion tags ([HAPPY], [SAD], [SHY], etc.) for expressive conversations
- **🎤 VOICEVOX Integration**: High-quality text-to-speech synthesis with multiple voice options
- **🎭 VTube Studio Support**: Control virtual avatars with hotkey triggers and expressions
- **🌐 Web Interface**: Modern Flask-based chat UI with real-time streaming responses
- **👥 Character Management**: Create and manage multiple AI personalities with custom system prompts
- **📝 CLI Chat Mode**: Direct command-line interface for interactive conversations
- **🔄 Streaming Responses**: Real-time character-by-character response generation

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, Flask, PyTorch
- **AI Model**: Qwen2.5-1.5B-Instruct with PEFT/LoRA fine-tuning
- **TTS Engine**: [VOICEVOX](https://voicevox.hiroshiba.jp/) (Local Engine, port 50021)
- **Avatar Control**: VTube Studio API integration (port 8001)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Libraries**: Transformers, Flask-CORS, WebSocket Client

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- PyTorch with CUDA support (recommended) or CPU
- [VOICEVOX Engine](https://voicevox.hiroshiba.jp/) for TTS
- [VTube Studio](https://denchisoft.com/) for avatar control (optional)

### 2. Model Setup
Ensure you have the following directories in the project:
```
Yuuna-Project/
├── Qwen2.5-1.5B-Instruct/  # Base model files
└── Lora/                   # LoRA adapter files
```

### 3. Installation
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Running the Application

**Web Interface Mode:**
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

**Command-Line Chat Mode:**
```bash
python chat.py
```

## 🎤 Voice Setup (VOICEVOX)

1. Download and install VOICEVOX
2. Launch VOICEVOX application (runs on port 50021)
3. Keep it running in the background
4. The web interface will automatically connect for TTS functionality

## 🎭 Avatar Setup (VTube Studio)

1. Install and launch VTube Studio
2. Enable API plugins in VTube Studio settings
3. Use the VTS test page at `http://localhost:5000/vts_test`
4. Configure hotkeys in VTube Studio for expression control

## 📁 Project Structure

```
Yuuna-Project/
├── app.py              # Main Flask web application
├── chat.py             # Command-line chat interface
├── vts_connector.py    # VTube Studio API connector
├── characters.json     # Character definitions storage
├── requirements.txt    # Python dependencies
├── static/            # CSS, JS, and image assets
├── templates/         # HTML templates
├── Qwen2.5-1.5B-Instruct/  # Base model directory
└── Lora/              # LoRA adapter directory
```

## 🎨 Character System

The project supports multiple AI characters:
- **Default Character**: Yuna-chan with predefined personality and emotion system
- **Custom Characters**: Create new personalities with custom system prompts
- **Character Management**: Add, edit, and delete characters through the web interface

## 🔧 Configuration

Key configuration options in `app.py`:
- `BASE_MODEL_PATH`: Path to Qwen model
- `LORA_PATH`: Path to LoRA adapter
- `VTS_HOST/VTS_PORT`: VTube Studio connection settings
- `VOICEVOX_URL`: VOICEVOX engine URL
- `DEFAULT_SPEAKER_ID`: Default voice for TTS

## 📄 License

This project is licensed under the MIT License. Feel free to explore and modify!

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and enhancement requests.