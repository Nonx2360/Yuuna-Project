import requests
import os

URL = "http://localhost:5000/api/tts"
STATUS_URL = "http://localhost:5000/api/tts/status"
OUTPUT_FILE = "test_output.wav"


def test_tts_status():
    try:
        response = requests.get(STATUS_URL, timeout=5)
        print("TTS status:", response.json())
    except Exception as e:
        print(f"Could not reach TTS status endpoint: {e}")


def test_tts():
    print("Testing Yuuna-Project Qwen3-TTS endpoint...")
    test_tts_status()

    payload = {
        "text": "Hello! I'm Yuuna-chan, nice to meet you!",
        "emotion": "cheerful",
        "intensity": 0.7,
    }

    try:
        response = requests.post(URL, json=payload, timeout=120)

        if response.status_code == 200:
            with open(OUTPUT_FILE, "wb") as f:
                f.write(response.content)
            print(f"Success! Audio saved to {OUTPUT_FILE}")
        elif response.status_code == 503:
            print("Failure: No TTS engine available.")
        else:
            print(f"Failure: Status {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_tts()
