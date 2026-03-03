# GPT-SoVITS Setup and Finetuning for Yuuna-Project

This guide will help you set up GPT-SoVITS for high-quality TTS and prepare your own custom voice models.

## 1. Installation

To get started, you'll need the GPT-SoVITS engine. Run the `setup_tts.ps1` script to automate the setup (cloning, venv, and model downloads):

```powershell
./setup_tts.ps1
```

> [!TIP]
> This script now automatically downloads the essential pretrained models (BERT, HuBERT, and base weights) using `download_models.py`.

## 2. Using the Base Model

The current implementation in `app.py` is configured to use the GPT-SoVITS API. 
1. Open a terminal in the `TTS/GPT-SoVITS` folder.
2. Ensure your virtual environment is active:
   ```powershell
   ./venv/Scripts/Activate
   ```
3. Run the API server:
   ```bash
   python api.py
   ```
4. Your Yuuna-Project will now send text to this API on port 9880.

## 3. How to Finetune Your Own Voice

Finetuning allows you to make the TTS sound exactly like a specific character.

### Prerequisites
- **Audio Data**: 1-2 hours of clean audio of the target voice (WAV format, 44.1kHz or 48kHz).
- **Transcripts**: Text files matching each audio clip.

### Process
1. **Data Processing**: Use the GPT-SoVITS WebUI (`python webui.py`) to slice and transcribe your audio.
2. **Training**: 
   - Go to the "Fine-tuning" tab in the WebUI.
   - Set the batch size and epochs (default is usually fine for a start).
   - Click "Start Training".
3. **Exporting**: Once finished, you'll get a `.pth` (SoVITS) and `.ckpt` (GPT) model.

### Using Your Custom Model
Once you have your finetuned model, update `app.py` or the API call to point to your new model files.

```python
# Example API update for custom model
payload = {
    "text": "Hello, I am using my custom voice!",
    "gpt_model_path": "path/to/your/model.ckpt",
    "sovits_model_path": "path/to/your/model.pth"
}
```
