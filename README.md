# Yuuna Project

Yuuna Project is a comprehensive AI companion system featuring Yuna-chan, a caring and expressive AI personality. This project combines a locally hosted language model with advanced integrations for voice synthesis, character management, and virtual avatar control.

## ✨ Features

- **🤖 Local AI Model**: Qwen2.5-1.5B-Instruct with custom LoRA fine-tuning for Yuna-chan's personality
- **💬 Emotion-Based Responses**: Yuna-chan responds with emotion tags ([HAPPY], [SAD], [SHY], etc.) for expressive conversations
- **🎤 VOICEVOX Integration**: High-quality text-to-speech synthesis with multiple voice options
- **🎭 VTube Studio Support**: Control virtual avatars with hotkey triggers and parameter injection
- **🌐 Web Interface**: Modern Flask-based chat UI with real-time streaming responses
- **👥 Character Management**: Create and manage multiple AI personalities with custom system prompts
- **📝 CLI Chat Mode**: Direct command-line interface for interactive conversations
- **🔄 Streaming Responses**: Real-time character-by-character response generation
- **⚡ 4-Bit Quantization**: Optional GPU memory optimization via BitsAndBytes

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, Flask, PyTorch
- **AI Model**: Qwen2.5-1.5B-Instruct with PEFT/LoRA fine-tuning
- **TTS Engine**: [VOICEVOX](https://voicevox.hiroshiba.jp/) (Local Engine, port 50021)
- **Avatar Control**: VTube Studio API integration (WebSocket, port 8001)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Libraries**: Transformers, Flask-CORS, WebSocket Client, PEFT, BitsAndBytes

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
└── Qwen25-lora-finetuned/  # LoRA adapter files
```

### 3. Installation
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

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
5. Map emotions to hotkeys via the web interface

## 📁 Project Structure

```
Yuuna-Project/
├── app.py                 # Main Flask web application
├── chat.py                # Command-line chat interface
├── vts_connector.py       # VTube Studio API connector
├── characters.json        # Character definitions storage
├── vts_mappings.json      # VTube Studio emotion-to-hotkey mappings
├── .vts_token             # VTube Studio authentication token
├── requirements.txt       # Python dependencies
├── static/                # CSS, JS, and image assets
│   ├── css/
│   └── js/
├── templates/             # HTML templates
│   ├── index.html         # Main chat interface
│   └── vts_test.html      # VTube Studio test page
├── Qwen2.5-1.5B-Instruct/ # Base model directory
├── Qwen25-lora-finetuned/ # LoRA adapter directory
└── dataset.jsonl          # Training dataset
```

## 🎨 Character System

The project supports multiple AI characters:
- **Default Character (Yuuna-chan)**: Uses LoRA fine-tuned model with emotion tags
- **Custom Characters**: Use base model with custom system prompts (no LoRA)
- **Character Management**: Add, edit, and delete characters through the web interface
- **System Prompt Generator**: AI-assisted prompt creation for custom characters

### Emotion Tags
Default character uses emotion tags: [HAPPY], [SAD], [SHY], [NOSTALGIC], [WORRIED], [LOVING], [CALM], [CURIOUS], [SURPRISED]

## 🔧 Configuration

Key configuration options in `app.py`:
- `BASE_MODEL_PATH`: Path to Qwen model directory
- `LORA_PATH`: Path to LoRA adapter directory
- `VTS_HOST/VTS_PORT`: VTube Studio connection settings (default: 127.0.0.1:8001)
- `VOICEVOX_URL`: VOICEVOX engine URL (default: http://localhost:50021)
- `DEFAULT_SPEAKER_ID`: Default voice ID for TTS (default: 2)

## 🎯 API Endpoints

### Chat
- `POST /api/chat` - Stream chat responses (SSE)

### TTS
- `POST /api/tts` - Generate voice audio (returns WAV)

### Characters
- `GET /api/characters` - List all characters
- `POST /api/characters` - Create new character
- `DELETE /api/characters/<id>` - Delete character

### VTube Studio
- `GET /api/vts/config` - Get VTS connection status
- `POST /api/vts/connect` - Authenticate with VTS
- `POST /api/vts/clear_token` - Clear stored token
- `GET /api/vts/hotkeys` - List available hotkeys
- `POST /api/vts/trigger` - Trigger hotkey by ID
- `POST /api/vts/parameter` - Inject parameter value
- `GET /api/vts/mapping` - Get emotion-hotkey mappings
- `POST /api/vts/mapping` - Save emotion-hotkey mappings

### Prompt Generation
- `POST /api/generate_prompt` - Generate system prompt from instruction

## 📄 License

This project is licensed under the MIT License. Feel free to explore and modify!

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and enhancement requests.